from pathlib import Path

import pytest

from katha import repo
from katha.stages import assemble
from katha.stages.types import AlignedWord, DiarizedEntry

# Speaker "1" recites briefly; speaker "0" explains at length. The airtime gap is
# what infer_reciter_speaker_id keys on, so keep it unambiguous — a tie would make
# the test pass on dict ordering rather than on the heuristic.
ENTRIES = [
    DiarizedEntry(speaker_id="1", text="ਤੋ ਸੋ ਨਹੀ", start_s=2.0, end_s=3.0),
    DiarizedEntry(speaker_id="0", text="ਐ ਪਾਤਸ਼ਾਹ", start_s=3.5, end_s=5.5),
]
WORDS = [
    AlignedWord(text="ਤੋ", start_s=2.10, end_s=2.30, score=0.90),
    AlignedWord(text="ਸੋ", start_s=2.31, end_s=2.50, score=0.80),
    AlignedWord(text="ਨਹੀ", start_s=2.51, end_s=2.90, score=0.70),   # -> 800 ms of reciter
    AlignedWord(text="ਐ", start_s=3.60, end_s=3.80, score=0.60),
    AlignedWord(text="ਪਾਤਸ਼ਾਹ", start_s=3.81, end_s=5.40, score=0.50),  # -> 1800 ms of kathakar
]


def test_alignment_text_joins_entries_in_order():
    assert assemble.alignment_text(ENTRIES) == "ਤੋ ਸੋ ਨਹੀ ਐ ਪਾਤਸ਼ਾਹ"


def test_assemble_takes_spans_from_the_aligned_words_not_the_asr():
    segments = assemble.assemble_segments(ENTRIES, WORDS)

    assert segments[0].start_ms == 2100   # first word's start, not the ASR's 2.0
    assert segments[0].end_ms == 2900     # last word's end, not the ASR's 3.0
    assert segments[0].timing_source == "aligned"
    # pytest.approx: sum([0.90, 0.80, 0.70]) / 3 == 0.7999999999999999 in binary
    # float, one ULP off exact 0.80 — a floating-point artifact, not a logic bug.
    assert segments[0].align_score == pytest.approx(0.80)  # mean of 0.90, 0.80, 0.70
    assert [w.text for w in segments[0].words] == ["ਤੋ", "ਸੋ", "ਨਹੀ"]
    assert segments[1].start_ms == 3600


def test_assemble_falls_back_to_asr_timings_when_word_counts_disagree():
    segments = assemble.assemble_segments(ENTRIES, WORDS[:-1])  # one word short

    assert [s.timing_source for s in segments] == ["asr", "asr"]
    assert segments[0].start_ms == 2000   # straight from the ASR chunk
    assert segments[0].end_ms == 3000
    assert segments[0].align_score is None
    assert segments[0].words == []


def test_infer_reciter_is_the_speaker_with_less_airtime():
    # The kathakar explains at length; the reciter only recites. Least airtime wins.
    segments = assemble.assemble_segments(ENTRIES, WORDS)
    assert assemble.infer_reciter_speaker_id(segments) == "1"


def test_infer_reciter_is_none_when_diarization_found_one_speaker():
    solo = [DiarizedEntry(speaker_id="0", text="ਜੀ", start_s=0.0, end_s=1.0)]
    segments = assemble.assemble_segments(solo, [
        AlignedWord(text="ਜੀ", start_s=0.0, end_s=1.0, score=0.9)
    ])
    assert assemble.infer_reciter_speaker_id(segments) is None


def test_role_for_maps_ids_to_roles():
    assert assemble.role_for("1", "1") == "reciter"
    assert assemble.role_for("0", "1") == "kathakar"
    assert assemble.role_for("0", None) == "unknown"


@pytest.mark.db
def test_save_segments_persists_roles_and_words(db):
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    segments = assemble.assemble_segments(ENTRIES, WORDS)
    repo.save_segments(db, clip_id, segments, reciter_id="1")

    with db.cursor() as cur:
        cur.execute(
            "SELECT idx, speaker_role, start_ms, text_pa FROM segments"
            " WHERE clip_id = %s ORDER BY idx",
            (clip_id,),
        )
        rows = cur.fetchall()
        cur.execute("SELECT count(*) AS n FROM words")
        word_count = cur.fetchone()["n"]

    assert [r["speaker_role"] for r in rows] == ["reciter", "kathakar"]
    assert rows[0]["start_ms"] == 2100
    assert word_count == 5
    assert repo.get_clip(db, clip_id)["reciter_speaker_id"] == "1"


@pytest.mark.db
def test_save_segments_replaces_a_previous_run(db):
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    segments = assemble.assemble_segments(ENTRIES, WORDS)
    repo.save_segments(db, clip_id, segments, reciter_id="1")
    repo.save_segments(db, clip_id, segments, reciter_id="1")  # a retry

    with db.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM segments WHERE clip_id = %s", (clip_id,))
        assert cur.fetchone()["n"] == 2
        # words cascade off segments on DELETE; if the cascade silently failed
        # this second save would leave 10 orphaned/doubled rows instead of 5.
        cur.execute("SELECT count(*) AS n FROM words")
        assert cur.fetchone()["n"] == 5


@pytest.mark.db
def test_save_segments_with_no_reciter_persists_unknown_role(db):
    # Sarvam's beta diarization can collapse to a single speaker, so
    # infer_reciter_speaker_id returns None and the caller passes
    # reciter_id=None through to save_segments. This is the realistic path
    # that stamps speaker_role='unknown' (which must satisfy the segments
    # table's speaker_role CHECK constraint) and clears reciter_speaker_id.
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    segments = assemble.assemble_segments(ENTRIES, WORDS)
    repo.save_segments(db, clip_id, segments, reciter_id=None)

    with db.cursor() as cur:
        cur.execute(
            "SELECT speaker_role FROM segments WHERE clip_id = %s ORDER BY idx",
            (clip_id,),
        )
        rows = cur.fetchall()

    assert [r["speaker_role"] for r in rows] == ["unknown", "unknown"]
    assert repo.get_clip(db, clip_id)["reciter_speaker_id"] is None
