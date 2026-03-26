from __future__ import annotations

from typing import Any, Iterator, Sequence


def chunked(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def column_prefix(property_key: str) -> str:
    return property_key.replace(".", "__")
