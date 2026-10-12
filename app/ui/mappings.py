"""
Qt-free slider <-> physical-value mappings.

Integer QSliders cannot represent floats or log scales directly; these small
immutable classes translate between slider positions and physical values and are
unit-testable without a display.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LinearMapping:
    """Slider position 0..steps  <->  value in [lo, hi] (linear)."""

    lo: float
    hi: float
    steps: int = 1000

    def to_value(self, pos: int) -> float:
        return self.lo + (self.hi - self.lo) * (pos / self.steps)

    def to_pos(self, value: float) -> int:
        frac = (min(max(value, self.lo), self.hi) - self.lo) / (self.hi - self.lo)
        return int(round(frac * self.steps))


@dataclass(frozen=True)
class LogMapping:
    """Slider position 0..steps  <->  value in [lo, hi] (logarithmic, lo > 0)."""

    lo: float
    hi: float
    steps: int = 1000

    def to_value(self, pos: int) -> float:
        a, b = math.log10(self.lo), math.log10(self.hi)
        return 10.0 ** (a + (b - a) * (pos / self.steps))

    def to_pos(self, value: float) -> int:
        a, b = math.log10(self.lo), math.log10(self.hi)
        v = min(max(value, self.lo), self.hi)
        return int(round((math.log10(v) - a) / (b - a) * self.steps))
