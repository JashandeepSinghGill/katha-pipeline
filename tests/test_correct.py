import json

import pytest

from katha.stages.correct import (
    FLAG_SCHEMA,
    ProposedFlag,
    SegmentText,
    build_system_prompt,
    build_user_prompt,
    flags_from_json,
    flags_to_json,
    make_windows,
    validate_flags,
)


def segs(n):
    return [
        SegmentText(idx=i, role="kathakar" if i % 2 else "reciter", text=f"ਸ਼ਬਦ{i} ਹੋਰ{i}")
        for i in range(n)
    ]


def test_windows_cover_every_segment_exactly_once_as_a_target():
    ws = make_windows(segs(10), size=4, context=1)
    assert [w.target_idx for w in ws] == [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9)]
    assert [w.index for w in ws] == [0, 1, 2]
    assert [s.idx for s in ws[0].segments] == [0, 1, 2, 3, 4]
    assert [s.idx for s in ws[1].segments] == [3, 4, 5, 6, 7, 8]
    assert [s.idx for s in ws[2].segments] == [7, 8, 9]


def test_windows_of_empty_input_is_empty():
    assert make_windows([], size=4, context=1) == []


def test_user_prompt_marks_targets_and_context_as_json_lines():
    w = make_windows(segs(3), size=2, context=1)[0]
    lines = [json.loads(l) for l in build_user_prompt(w).splitlines() if l.startswith("{")]
    assert [(l["idx"], l["target"]) for l in lines] == [(0, True), (1, True), (2, False)]
    assert lines[0]["text"] == "ਸ਼ਬਦ0 ਹੋਰ0"
    assert lines[0]["role"] == "reciter"


def test_system_prompt_includes_gazetteer_and_rules():
    p = build_system_prompt([("ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", "person"), ("ਤਲਵੰਡੀ", "place")])
    assert "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ" in p and "(person)" in p
    assert "verbatim" in p.lower()
    assert "Gurmukhi" in p
    assert build_system_prompt([]).strip()


def test_schema_is_strict_and_bounded():
    item = FLAG_SCHEMA["properties"]["flags"]["items"]
    assert FLAG_SCHEMA["additionalProperties"] is False
    assert item["additionalProperties"] is False
    assert item["properties"]["candidates"]["maxItems"] == 5
    assert item["properties"]["candidates"]["minItems"] == 1
    assert set(item["required"]) == {"segment_idx", "span_text", "kind", "reason", "candidates"}


def _window():
    return make_windows(segs(3), size=2, context=1)[0]  # targets 0,1; context 2


def test_validate_keeps_good_flags_and_normalises_candidates():
    payload = {"flags": [{
        "segment_idx": 1, "span_text": " ਹੋਰ1 ", "kind": "proper_noun", "reason": "r",
        "candidates": [" ਹੋਰ੧ ", "ਹੋਰ੧", "ਹੋਰ1", "", "ਹੋਰ੨"],
    }]}
    flags, dropped = validate_flags(payload, _window())
    assert dropped == 0
    assert flags == [ProposedFlag(1, "ਹੋਰ1", "proper_noun", "r", ("ਹੋਰ੧", "ਹੋਰ੨"))]


@pytest.mark.parametrize("bad", [
    {"segment_idx": 2, "span_text": "ਹੋਰ2", "kind": "other", "reason": "context-only", "candidates": ["x"]},
    {"segment_idx": 9, "span_text": "ਹੋਰ1", "kind": "other", "reason": "unknown idx", "candidates": ["x"]},
    {"segment_idx": 1, "span_text": "ਨਹੀਂ", "kind": "other", "reason": "not verbatim", "candidates": ["x"]},
    {"segment_idx": 1, "span_text": "", "kind": "other", "reason": "empty span", "candidates": ["x"]},
    {"segment_idx": 1, "span_text": "ਹੋਰ1", "kind": "other", "reason": "no candidate", "candidates": ["ਹੋਰ1", " "]},
    {"segment_idx": "1", "span_text": "ਹੋਰ1", "kind": "other", "reason": "idx not int", "candidates": ["x"]},
    {"segment_idx": True, "span_text": "ਹੋਰ1", "kind": "other", "reason": "bool idx", "candidates": ["x"]},
    "not an object",
])
def test_validate_drops_invalid_items(bad):
    flags, dropped = validate_flags({"flags": [bad]}, _window())
    assert flags == [] and dropped == 1


def test_validate_truncates_to_five_candidates_and_defaults_unknown_kind():
    payload = {"flags": [{"segment_idx": 0, "span_text": "ਸ਼ਬਦ0", "kind": "weird", "reason": "",
                          "candidates": [f"c{i}" for i in range(8)]}]}
    flags, _ = validate_flags(payload, _window())
    assert flags[0].kind == "other"
    assert flags[0].candidates == ("c0", "c1", "c2", "c3", "c4")


def test_validate_tolerates_missing_or_malformed_flags_key():
    assert validate_flags({}, _window()) == ([], 0)
    assert validate_flags({"flags": "nope"}, _window()) == ([], 0)
    assert validate_flags("junk", _window()) == ([], 0)


def test_json_round_trip():
    flags = [ProposedFlag(1, "ਹੋਰ1", "proper_noun", "r", ("ਹੋਰ੧",))]
    assert flags_from_json(flags_to_json(flags)) == flags
