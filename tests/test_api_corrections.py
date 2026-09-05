from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from katha import repo
from katha.api.main import app, get_app_settings, get_conn
from katha.stages import assemble
from katha.stages.types import AlignedWord, DiarizedEntry

ENTRIES = [DiarizedEntry("0", "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ ਨੇ ਲਿਖਿਆ", 3.5, 5.5)]
WORDS = [AlignedWord(w, 3.5 + i * 0.2, 3.6 + i * 0.2, 0.9)
         for i, w in enumerate(ENTRIES[0].text.split())]


@pytest.fixture
def api(db, settings):
    app.dependency_overrides[get_conn] = lambda: db
    app.dependency_overrides[get_app_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def flagged(db):
    cid = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    repo.save_segments(db, cid, assemble.assemble_segments(ENTRIES, WORDS), reciter_id=None)
    sid = repo.list_segment_texts(db, cid)[0]["id"]
    repo.save_flags(db, cid, [(sid, "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", "proper_noun", "r", ["ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"])])
    return cid, repo.list_open_flags(db, cid)[0]["id"]


@pytest.mark.db
def test_clip_payload_includes_flags_and_list_counts(api, flagged):
    cid, fid = flagged
    body = api.get(f"/api/clips/{cid}").json()
    f = body["segments"][0]["flags"][0]
    assert f["id"] == fid and f["status"] == "open" and f["candidates"] == ["ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"]
    assert body["segments"][0]["corrections"] == []
    row = api.get("/api/clips").json()["clips"][0]
    assert row["open_flags"] == 1 and row["auto_applied"] == 0


@pytest.mark.db
def test_resolve_accept_then_revert_round_trip(api, flagged):
    cid, fid = flagged
    r = api.post(f"/api/flags/{fid}/resolve", json={
        "action": "accept", "text": "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ",
        "remember": {"entity_type": "person", "english_gloss": "poet"}})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["text_pa"] == "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ ਜੀ ਨੇ ਲਿਖਿਆ" and out["ledger_id"]
    seg = api.get(f"/api/clips/{cid}").json()["segments"][0]
    assert seg["flags"][0]["status"] == "accepted"
    assert seg["corrections"][0]["source"] == "admin"
    assert seg["corrections"][0]["reverted"] is False
    r = api.post(f"/api/corrections/{out['correction_id']}/revert")
    assert r.status_code == 200
    assert r.json()["text_pa"] == "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ ਨੇ ਲਿਖਿਆ"
    seg = api.get(f"/api/clips/{cid}").json()["segments"][0]
    assert seg["flags"][0]["status"] == "open" and seg["corrections"][0]["reverted"] is True
    assert api.post(f"/api/corrections/{out['correction_id']}/revert").status_code == 409
    assert api.post("/api/corrections/999999/revert").status_code == 404


@pytest.mark.db
def test_resolve_reject_and_error_statuses(api, flagged):
    cid, fid = flagged
    assert api.post(f"/api/flags/{fid}/resolve", json={"action": "accept"}).status_code == 400
    assert api.post(f"/api/flags/{fid}/resolve", json={"action": "nope"}).status_code == 422
    assert api.post(f"/api/flags/{fid}/resolve", json={"action": "reject"}).status_code == 200
    assert api.post(f"/api/flags/{fid}/resolve", json={"action": "reject"}).status_code == 409
    assert api.post("/api/flags/999999/resolve", json={"action": "reject"}).status_code == 404


@pytest.mark.db
def test_ledger_endpoints(api):
    r = api.post("/api/ledger", json={"canonical_gurmukhi": "ਤਲਵੰਡੀ",
                                      "surface_forms": ["ਤਲਬੰਡੀ"], "entity_type": "place"})
    assert r.status_code == 200, r.text
    lid = r.json()["id"]
    rows = api.get("/api/ledger", params={"q": "talaw"}).json()["entries"]  # ਵ romanizes to w
    assert [e["id"] for e in rows] == [lid]
    assert rows[0]["romanized"].startswith("tal")
    assert rows[0]["confidence_threshold"] == pytest.approx(0.8)
    assert rows[0]["surface_forms"] == ["ਤਲਬੰਡੀ"]
    r = api.patch(f"/api/ledger/{lid}", json={"surface_forms": ["ਤਲਬੰਡੀ", "ਤਲਵੰਢੀ"],
                                              "confidence_threshold": 0.9, "retired": True})
    assert r.status_code == 200
    assert api.get("/api/ledger").json()["entries"] == []
    row = api.get("/api/ledger", params={"include_retired": "true"}).json()["entries"][0]
    assert row["retired"] is True and row["surface_forms"] == ["ਤਲਬੰਡੀ", "ਤਲਵੰਢੀ"]
    assert row["confidence_threshold"] == pytest.approx(0.9)
    assert api.patch("/api/ledger/999999", json={"retired": False}).status_code == 404
    assert api.patch(f"/api/ledger/{lid}", json={}).status_code == 400
    assert api.patch(f"/api/ledger/{lid}", json={"confidence_threshold": 7}).status_code == 400
    assert api.post("/api/ledger", json={"canonical_gurmukhi": " ",
                                         "entity_type": "place"}).status_code == 400
    assert api.post("/api/ledger", json={"canonical_gurmukhi": "x",
                                         "entity_type": "alien"}).status_code == 422
    # re-posting the same canonical form merges surface forms instead of erroring
    r = api.post("/api/ledger", json={"canonical_gurmukhi": "ਤਲਵੰਡੀ",
                                      "surface_forms": ["ਤਲਵੰਢੀ", "ਤਲਵੰਡਾ"], "entity_type": "place"})
    assert r.status_code == 200 and r.json()["id"] == lid
    row = api.get("/api/ledger").json()["entries"][0]   # revived by the upsert
    assert row["surface_forms"] == ["ਤਲਬੰਡੀ", "ਤਲਵੰਢੀ", "ਤਲਵੰਡਾ"]


@pytest.mark.db
def test_settings_endpoints_validate(api):
    body = api.get("/api/settings").json()
    assert body["values"]["files_per_day"] == "3"
    assert any(s["key"] == "llm_model" for s in body["specs"])
    r = api.put("/api/settings", json={"values": {"files_per_day": "5", "auto_apply_enabled": "no"}})
    assert r.status_code == 200
    assert api.get("/api/settings").json()["values"]["auto_apply_enabled"] == "false"
    assert api.get("/api/settings").json()["values"]["files_per_day"] == "5"
    assert api.put("/api/settings", json={"values": {"files_per_day": "-2"}}).status_code == 400
    assert api.put("/api/settings", json={"values": {"bogus": "1"}}).status_code == 400
    assert api.put("/api/settings", json={"values": {"files_per_day": "9", "bogus": "1"}}).status_code == 400
    assert api.get("/api/settings").json()["values"]["files_per_day"] == "5"  # nothing partial
