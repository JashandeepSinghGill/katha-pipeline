import json
from pathlib import Path

import pytest

from katha.stages import asr

# Response shape verified against Sarvam's OpenAPI schema
# (Sarvam_Model_API_SpeechToTextResponse).
PAYLOAD = {
    "request_id": "req-1",
    "transcript": "ਕਵਿਤ। ਤੋ ਸੋ ਨਹੀ ਦਾਤਾ ਕੋਇ ॥ ਐ ਪਾਤਸ਼ਾਹ ਜੀ।",
    "language_code": "pa-IN",
    "diarized_transcript": {
        "entries": [
            {"transcript": "ਕਵਿਤ।", "start_time_seconds": 2.25,
             "end_time_seconds": 2.28, "speaker_id": "1"},
            {"transcript": "ਤੋ ਸੋ ਨਹੀ ਦਾਤਾ ਕੋਇ ॥", "start_time_seconds": 2.29,
             "end_time_seconds": 2.34, "speaker_id": "1"},
            {"transcript": "  ", "start_time_seconds": 2.35,
             "end_time_seconds": 2.36, "speaker_id": "0"},
            {"transcript": "ਐ ਪਾਤਸ਼ਾਹ ਜੀ।", "start_time_seconds": 2.41,
             "end_time_seconds": 2.44, "speaker_id": "0"},
        ]
    },
}


def test_parse_diarized_reads_entries_in_order():
    entries = asr.parse_diarized(PAYLOAD)
    assert [e.speaker_id for e in entries] == ["1", "1", "0"]
    assert entries[1].text == "ਤੋ ਸੋ ਨਹੀ ਦਾਤਾ ਕੋਇ ॥"
    assert entries[0].start_s == 2.25


def test_parse_diarized_drops_blank_entries():
    assert len(asr.parse_diarized(PAYLOAD)) == 3


def test_parse_diarized_coerces_speaker_id_to_str():
    # Sarvam's docs disagree with themselves: schema says "0"/"1", one page says SPEAKER_00.
    payload = {"diarized_transcript": {"entries": [
        {"transcript": "ਜੀ", "start_time_seconds": 0.0,
         "end_time_seconds": 1.0, "speaker_id": 0},
    ]}}
    assert asr.parse_diarized(payload)[0].speaker_id == "0"


def test_parse_diarized_without_diarization_is_an_error():
    with pytest.raises(asr.AsrError, match="diarized_transcript"):
        asr.parse_diarized({"transcript": "x"})


def test_parse_diarized_raises_asr_error_on_missing_timestamp_key():
    # A flaky/partial API response omitting a timestamp key must surface as
    # AsrError, not a bare KeyError callers aren't told to expect.
    payload = {"diarized_transcript": {"entries": [
        {"transcript": "ਜੀ", "end_time_seconds": 1.0, "speaker_id": "0"},
    ]}}
    with pytest.raises(asr.AsrError, match="entry 0.*start_time_seconds"):
        asr.parse_diarized(payload)


def test_parse_diarized_raises_asr_error_on_non_numeric_timestamp():
    payload = {"diarized_transcript": {"entries": [
        {"transcript": "ਜੀ", "start_time_seconds": "abc",
         "end_time_seconds": 1.0, "speaker_id": "0"},
    ]}}
    with pytest.raises(asr.AsrError, match="entry 0.*start_time_seconds"):
        asr.parse_diarized(payload)


class FakeJob:
    job_id = "job-123"

    def __init__(self, out_payload, out_filename="0.json"):
        self._payload = out_payload
        self._out_filename = out_filename
        self.started = False

    def upload_files(self, file_paths):
        self.uploaded = file_paths

    def start(self):
        self.started = True

    def wait_until_complete(self, poll_interval, timeout):
        self.waited = (poll_interval, timeout)

    def get_file_results(self):
        return {"successful": [self._out_filename], "failed": []}

    def download_outputs(self, output_dir):
        (Path(output_dir) / self._out_filename).write_text(
            json.dumps(self._payload), encoding="utf-8"
        )


class FakeClient:
    def __init__(self, payload, out_filename="0.json"):
        self.payload = payload
        self.out_filename = out_filename
        self.speech_to_text_job = self
        self.job = None

    def create_job(self, **kwargs):
        self.create_kwargs = kwargs
        self.job = FakeJob(self.payload, out_filename=self.out_filename)
        return self.job


def test_transcribe_pins_saaras_v3_and_a_long_timeout(tmp_path, settings):
    # The SDK's wait_until_complete() defaults to timeout=600s, which will
    # raise TimeoutError on a 90-minute katha clip. Guard the real fix
    # (settings.sarvam_timeout_seconds) rather than restating its value.
    assert settings.sarvam_timeout_seconds >= 7200

    client = FakeClient(PAYLOAD)
    entries, job_id = asr.SarvamAsr(settings, client=client).transcribe(
        Path("/audio/asr.wav"), tmp_path
    )

    assert client.create_kwargs["model"] == "saaras:v3"
    assert client.create_kwargs["language_code"] == "pa-IN"
    assert client.create_kwargs["with_diarization"] is True
    assert client.create_kwargs["with_timestamps"] is True
    assert client.create_kwargs["num_speakers"] == 2
    assert len(entries) == 3
    assert job_id == "job-123"
    assert client.job.waited == (
        settings.sarvam_poll_seconds,
        settings.sarvam_timeout_seconds,
    )


def test_transcribe_ignores_stale_json_left_in_out_dir(tmp_path, settings):
    # Task 7 retries reuse the same per-clip out_dir. A JSON left behind by a
    # previous attempt must not be picked up by sorted(glob(...))[0] instead
    # of this run's fresh output.
    stale_payload = {"diarized_transcript": {"entries": [
        {"transcript": "stale transcript", "start_time_seconds": 0.0,
         "end_time_seconds": 1.0, "speaker_id": "0"},
    ]}}
    (tmp_path / "0.json").write_text(json.dumps(stale_payload), encoding="utf-8")

    # The fresh job writes to a different (alphabetically later) filename than
    # the stale leftover, so that absent the fix, sorted(glob(...))[0] would
    # still pick the stale "0.json" instead of this run's "1.json".
    client = FakeClient(PAYLOAD, out_filename="1.json")
    entries, job_id = asr.SarvamAsr(settings, client=client).transcribe(
        Path("/audio/asr.wav"), tmp_path
    )

    assert [e.text for e in entries] == [
        "ਕਵਿਤ।", "ਤੋ ਸੋ ਨਹੀ ਦਾਤਾ ਕੋਇ ॥", "ਐ ਪਾਤਸ਼ਾਹ ਜੀ।",
    ]
    assert job_id == "job-123"
