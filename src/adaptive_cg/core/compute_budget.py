"""Compute-aware budget controller for adaptive CG simulations.

Monitors wall-clock time per step and adjusts total bead count
to maintain a target step rate. More compute available -> more beads
(higher resolution). Less compute -> fewer beads (stay real-time).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class ComputeBudget:
    """Tracks step throughput and recommends bead counts to hit a target rate.

    The key insight: non-bonded pair interactions scale as O(n^2), so if
    the measured rate is R and the target is T, we can afford
    n_new = n_current * sqrt(R / T) beads.
    """

    target_steps_per_second: float  # desired step throughput
    min_beads: int = 20             # absolute minimum beads
    max_beads: int = 500            # absolute maximum beads
    current_beads: int = 100        # current bead count
    adjustment_interval: int = 1000  # steps between budget checks
    smoothing: float = 0.3          # EMA smoothing for step rate measurement

    # Internal state
    _step_times: deque = field(default_factory=lambda: deque(maxlen=200))
    _measured_rate: float = 0.0
    _last_adjustment_step: int = 0

    def record_step(self, elapsed_seconds: float) -> None:
        """Record how long a single step took.

        Updates the exponentially weighted moving average of the step rate.
        """
        if elapsed_seconds <= 0:
            return

        self._step_times.append(elapsed_seconds)
        instantaneous_rate = 1.0 / elapsed_seconds

        if self._measured_rate == 0.0:
            # First measurement — seed the EMA.
            self._measured_rate = instantaneous_rate
        else:
            alpha = self.smoothing
            self._measured_rate = (
                alpha * instantaneous_rate + (1 - alpha) * self._measured_rate
            )

    @property
    def measured_rate(self) -> float:
        """Current smoothed step rate (steps/second)."""
        return self._measured_rate

    def should_adjust(self, current_step: int) -> bool:
        """Check if it's time to adjust bead count."""
        if not self._step_times:
            return False
        return (
            current_step - self._last_adjustment_step >= self.adjustment_interval
        )

    def recommend_beads(self) -> int:
        """Recommend new bead count based on measured performance.

        If measured_rate > target * 1.2: increase beads (have headroom).
        If measured_rate < target * 0.8: decrease beads (too slow).
        Otherwise keep current count (inside dead zone).

        Scaling: non-bonded cost ~ n^2, so affordable bead count scales
        as n_new = n_current * sqrt(measured_rate / target_rate).
        Result is clamped to [min_beads, max_beads].
        """
        if self._measured_rate <= 0:
            return self.current_beads

        ratio = self._measured_rate / self.target_steps_per_second

        # Dead zone: don't adjust if within 20% of target.
        if 0.8 <= ratio <= 1.2:
            return self.current_beads

        # n^2 scaling: if we can do sqrt(ratio) times more pairs, we can
        # afford sqrt(ratio) times more beads.
        scale = math.sqrt(ratio)
        new_beads = int(round(self.current_beads * scale))
        return max(self.min_beads, min(self.max_beads, new_beads))

    def status(self) -> str:
        """Human-readable status string."""
        if self._measured_rate <= 0:
            return (
                f"ComputeBudget: {self.current_beads} beads, "
                f"target={self.target_steps_per_second:.0f} steps/s, "
                f"no measurements yet"
            )
        ratio = self._measured_rate / self.target_steps_per_second
        if ratio > 1.2:
            state = "headroom"
        elif ratio < 0.8:
            state = "overloaded"
        else:
            state = "on-target"
        return (
            f"ComputeBudget: {self.current_beads} beads, "
            f"measured={self._measured_rate:.1f} steps/s, "
            f"target={self.target_steps_per_second:.0f} steps/s, "
            f"state={state}"
        )


def auto_configure(
    n_atoms: int,
    target_steps_per_second: float | None = None,
    hardware_pairs_per_second: float | None = None,
) -> ComputeBudget:
    """Create a ComputeBudget with sensible defaults.

    Parameters
    ----------
    n_atoms:
        Total atoms in the all-atom system (used to set bead range).
    target_steps_per_second:
        Desired throughput. Defaults to 100 (interactive).
    hardware_pairs_per_second:
        If known, the number of pairwise interactions per second this
        hardware can evaluate. Used to estimate a good starting bead
        count. Otherwise start with n_atoms // 4 and let the controller
        adapt.
    """
    if target_steps_per_second is None:
        target_steps_per_second = 100.0

    # Bead limits scale with system size.
    min_beads = max(10, n_atoms // 20)
    max_beads = max(min_beads + 1, n_atoms // 2)

    if hardware_pairs_per_second is not None and hardware_pairs_per_second > 0:
        # Each step evaluates ~n*(n-1)/2 pairs. Solve for n:
        #   pairs_per_step = n*(n-1)/2
        #   pairs_per_step * target_rate <= hardware_pairs_per_second
        #   n*(n-1)/2 <= hardware_pairs_per_second / target_rate
        #   n ~ sqrt(2 * budget)
        pair_budget = hardware_pairs_per_second / target_steps_per_second
        initial_beads = int(math.sqrt(2.0 * pair_budget))
    else:
        initial_beads = max(min_beads, n_atoms // 4)

    initial_beads = max(min_beads, min(max_beads, initial_beads))

    return ComputeBudget(
        target_steps_per_second=target_steps_per_second,
        min_beads=min_beads,
        max_beads=max_beads,
        current_beads=initial_beads,
    )
