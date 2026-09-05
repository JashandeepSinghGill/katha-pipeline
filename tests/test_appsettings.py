import pytest

from katha import appsettings


def test_every_default_round_trips_through_validate():
    for key, default in appsettings.DEFAULTS.items():
        assert appsettings.validate(key, default) == default


def test_coerce_returns_typed_values():
    assert appsettings.coerce("files_per_day", "3") == 3
    assert appsettings.coerce("ledger_context_threshold", "0.8") == 0.8
    assert appsettings.coerce("auto_apply_enabled", "true") is True
    assert appsettings.coerce("auto_apply_enabled", "false") is False
    assert appsettings.coerce("llm_model", "claude-sonnet-5") == "claude-sonnet-5"


def test_validate_rejects_out_of_range_and_wrong_type():
    with pytest.raises(ValueError):
        appsettings.validate("files_per_day", "-1")
    with pytest.raises(ValueError):
        appsettings.validate("ledger_context_threshold", "1.5")
    with pytest.raises(ValueError):
        appsettings.validate("ledger_surface_threshold", "abc")
    with pytest.raises(ValueError):
        appsettings.validate("auto_apply_enabled", "maybe")
    with pytest.raises(ValueError):
        appsettings.validate("llm_model", "   ")


def test_validate_canonicalises_bools_and_rejects_unknown_keys():
    assert appsettings.validate("auto_apply_enabled", "YES") == "true"
    assert appsettings.validate("auto_apply_enabled", "0") == "false"
    with pytest.raises(KeyError):
        appsettings.validate("no_such_key", "1")
