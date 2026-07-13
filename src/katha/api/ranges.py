import re

_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """RFC 7233 single-range parsing. Returns inclusive (start, end), or None for
    'send the whole file'. Raises ValueError when the range cannot be satisfied."""
    if not header:
        return None
    match = _RANGE.fullmatch(header.strip())
    if not match:
        return None

    raw_start, raw_end = match.group(1), match.group(2)
    if not raw_start and not raw_end:
        return None

    if not raw_start:  # suffix range: last N bytes
        length = int(raw_end)
        if length <= 0:
            raise ValueError("zero-length suffix range")
        return max(0, size - length), size - 1

    start = int(raw_start)
    end = int(raw_end) if raw_end else size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        raise ValueError(f"range {start}-{end} is outside a {size}-byte file")
    return start, end
