"""Shared result types for the recycler modules.

Every module returns a ``ModuleResult`` with a one-sentence ``plain_summary``
(no jargon) and ``key_facts`` that the UI renders directly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KeyFact:
    """One row of the "key facts" grid."""

    label: str
    value: str             # already formatted for display, e.g. "1.36 +/- 0.05"
    unit: str = ""
    tooltip: str = ""

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.label, self.value, self.unit, self.tooltip)


@dataclass
class ModuleResult:
    module: str                                    # planet | eb | variable | noise
    plain_summary: str
    key_facts: list[KeyFact] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["key_facts"] = [k.as_tuple() for k in self.key_facts]
        return d


def fmt(x: float, digits: int = 2) -> str:
    """A number for display, or "unknown"."""
    if x is None or not math.isfinite(x):
        return "unknown"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


def fmt_pm(median: float, lo: float, hi: float, digits: int = 2) -> str:
    """"median +/- error" (or "+a / -b" when the error is lopsided)."""
    if not all(v is not None and math.isfinite(v) for v in (median, lo, hi)):
        return fmt(median, digits)
    up, down = hi - median, median - lo
    if max(up, down) <= 0:
        return fmt(median, digits)
    if abs(up - down) <= 0.2 * max(up, down):
        return f"{fmt(median, digits)} +/- {fmt((up + down) / 2, digits)}"
    return f"{fmt(median, digits)} (+{fmt(up, digits)} / -{fmt(down, digits)})"


def period_words(days: float) -> str:
    """'3.5 days', '15 hours'."""
    if not math.isfinite(days):
        return "an unknown time"
    if days < 1:
        return f"{days * 24:.0f} hours"
    return f"{days:.1f} days" if days < 100 else f"{days:.0f} days"
