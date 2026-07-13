from pathlib import Path
from katha.config import Settings


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("KATHA_SARVAM_API_KEY", "sk-test")
    monkeypatch.setenv("KATHA_WATCH_DIR", "/tmp/in")
    s = Settings(_env_file=None)
    assert s.sarvam_api_key == "sk-test"
    assert s.watch_dir == Path("/tmp/in")


def test_sarvam_defaults_are_the_verified_values():
    s = Settings(_env_file=None)
    assert s.sarvam_model == "saaras:v3"      # SDK default is the legacy saarika:v2.5
    assert s.sarvam_language == "pa-IN"       # Sarvam uses ISO-639-1 + region
    assert s.align_language == "pan"          # aligner uses ISO-639-3
    assert s.sarvam_timeout_seconds >= 7200   # wait_until_complete default of 600s is too short
