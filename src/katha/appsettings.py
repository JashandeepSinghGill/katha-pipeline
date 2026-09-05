"""Typed specs for the admin-editable rows in the `settings` table.

The migration seeds these rows (values duplicated there on purpose — a
migration must never import application code, or it drifts); a test asserts
the seed equals DEFAULTS so the two can't silently diverge.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SettingSpec:
    key: str
    kind: str  # 'int' | 'float' | 'bool' | 'str'
    default: str
    minimum: float | None = None
    maximum: float | None = None
    help: str = ""


SPECS: dict[str, SettingSpec] = {
    s.key: s
    for s in (
        SettingSpec("files_per_day", "int", "3", 0, 1000,
                    "New clips the worker may start per rolling 24 h."),
        SettingSpec("llm_model", "str", "claude-sonnet-5", help=
                    "Model id for the correction pass (full id, e.g. claude-sonnet-5)."),
        SettingSpec("correct_window_segments", "int", "40", 1, 500,
                    "Segments sent to the LLM per call."),
        SettingSpec("correct_context_segments", "int", "3", 0, 50,
                    "Neighbouring segments shown as context on each side of a window."),
        SettingSpec("ledger_surface_threshold", "float", "88.0", 0, 100,
                    "Minimum rapidfuzz ratio (0-100) between a flagged span and a ledger surface form."),
        SettingSpec("ledger_context_threshold", "float", "0.8", 0, 1,
                    "Default cosine threshold stored on new ledger entries for auto-apply."),
        SettingSpec("auto_apply_enabled", "bool", "true", help=
                    "When off, ledger matches are only suggested, never applied."),
    )
}
DEFAULTS: dict[str, str] = {k: s.default for k, s in SPECS.items()}

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def coerce(key: str, raw: str) -> int | float | bool | str:
    spec = SPECS[key]
    raw = raw.strip()
    if spec.kind == "int":
        return int(raw)
    if spec.kind == "float":
        return float(raw)
    if spec.kind == "bool":
        low = raw.lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if not raw:
        raise ValueError("must not be empty")
    return raw


def validate(key: str, value: str) -> str:
    """Return the canonical string to store, or raise KeyError / ValueError."""
    spec = SPECS[key]  # KeyError for unknown keys is intentional
    try:
        typed = coerce(key, value)
    except ValueError as exc:
        raise ValueError(f"{key}: {exc}") from None
    if spec.kind in ("int", "float"):
        if spec.minimum is not None and typed < spec.minimum:
            raise ValueError(f"{key}: must be >= {spec.minimum}")
        if spec.maximum is not None and typed > spec.maximum:
            raise ValueError(f"{key}: must be <= {spec.maximum}")
        return str(typed) if spec.kind == "int" else repr(float(typed))
    if spec.kind == "bool":
        return "true" if typed else "false"
    return typed
