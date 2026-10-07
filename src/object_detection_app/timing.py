"""Bounded rolling timing samples; never retain frames or per-frame history forever."""

from collections import deque

import numpy as np


class Timings:
    def __init__(self, capacity: int = 300):
        self.samples = deque(maxlen=capacity)

    def add(self, **seconds: float) -> None:
        self.samples.append(seconds)

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0}
        result = {"samples": len(self.samples)}
        for key in self.samples[0]:
            values = [sample[key] * 1000 for sample in self.samples]
            result[key + "_ms"] = {
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
            }
        return result

    @property
    def fps(self) -> float:
        if not self.samples:
            return 0.0
        duration = sum(s["total"] for s in self.samples)
        return len(self.samples) / duration if duration else 0.0
