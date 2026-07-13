import pytest

from katha.api.ranges import parse_range


def test_no_header_means_the_whole_file():
    assert parse_range(None, 1000) is None


def test_open_ended_range_runs_to_the_last_byte():
    assert parse_range("bytes=500-", 1000) == (500, 999)


def test_closed_range_is_inclusive():
    assert parse_range("bytes=0-499", 1000) == (0, 499)


def test_range_past_the_end_is_clamped():
    assert parse_range("bytes=900-5000", 1000) == (900, 999)


def test_suffix_range_counts_back_from_the_end():
    assert parse_range("bytes=-200", 1000) == (800, 999)


def test_a_start_past_the_end_is_unsatisfiable():
    with pytest.raises(ValueError):
        parse_range("bytes=2000-", 1000)


def test_a_malformed_header_is_ignored_rather_than_fatal():
    assert parse_range("furlongs=0-10", 1000) is None
