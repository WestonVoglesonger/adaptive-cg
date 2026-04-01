"""Hardware detection and compute estimation for adaptive resolution."""
from __future__ import annotations

import math
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass

import numpy as np

# Try to import Rust backend for accurate benchmarking
try:
    import cg_engine as _rs
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass
class HardwareProfile:
    cpu_cores: int
    cpu_name: str
    gpu_available: bool
    gpu_name: str | None
    memory_gb: float
    estimated_pairs_per_second: float  # non-bonded pair evaluations per second


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------

def _detect_cpu_name() -> str:
    """Return a human-readable CPU model string."""
    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return out.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_gpu_macos() -> str | None:
    """Detect Metal GPU on macOS via system_profiler."""
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        # Look for "Chipset Model:" or "Chip:" lines
        for line in out.splitlines():
            stripped = line.strip()
            if stripped.startswith("Chipset Model:") or stripped.startswith("Chip:"):
                return stripped.split(":", 1)[1].strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_gpu_linux() -> str | None:
    """Detect NVIDIA GPU via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        name = out.strip().splitlines()[0].strip()
        if name:
            return name
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired, IndexError):
        pass
    return None


def _detect_gpu() -> tuple[bool, str | None]:
    """Detect GPU availability and name.

    Returns (gpu_available, gpu_name).
    """
    system = platform.system()
    if system == "Darwin":
        name = _detect_gpu_macos()
        # Apple Silicon always has Metal; discrete GPUs show up too
        return (name is not None, name)
    elif system == "Linux":
        name = _detect_gpu_linux()
        return (name is not None, name)
    return (False, None)


# ---------------------------------------------------------------------------
# Memory detection
# ---------------------------------------------------------------------------

def _detect_memory_gb() -> float:
    """Detect total system memory in GB."""
    system = platform.system()

    # macOS: sysctl hw.memsize (bytes)
    if system == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return int(out.strip()) / (1024 ** 3)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass

    # Linux: os.sysconf
    if system == "Linux":
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return (pages * page_size) / (1024 ** 3)
        except (ValueError, OSError):
            pass
        # Fallback: /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        m = re.search(r"(\d+)", line)
                        if m:
                            return int(m.group(1)) / (1024 ** 2)  # kB → GB
        except OSError:
            pass

    # Last resort: assume 8 GB
    return 8.0


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------

def _benchmark_pairs_per_second(n_beads: int = 100, n_iters: int = 1000) -> float:
    """Run a quick LJ force benchmark and return estimated pair evaluations/s.

    Uses the Rust backend if available, otherwise falls back to vectorized
    NumPy pairwise computation with a conservative estimate.
    """
    rng = np.random.default_rng(42)
    positions = rng.uniform(0.0, 3.0, size=(n_beads, 3))
    n_pairs = n_beads * (n_beads - 1) // 2

    if _HAS_RUST:
        # Build all-pairs list for Rust LJ kernel
        pair_i = []
        pair_j = []
        for i in range(n_beads):
            for j in range(i + 1, n_beads):
                pair_i.append(i)
                pair_j.append(j)
        pair_i_arr = np.array(pair_i, dtype=np.int64)
        pair_j_arr = np.array(pair_j, dtype=np.int64)
        sigma_arr = np.full(n_pairs, 0.4, dtype=np.float64)
        epsilon_arr = np.full(n_pairs, 1.0, dtype=np.float64)

        # Empty bond/angle/dihedral arrays
        empty_i32 = np.array([], dtype=np.int64)
        empty_f64 = np.array([], dtype=np.float64)

        # Warm up
        _rs.compute_all_forces(
            positions,
            empty_i32, empty_i32, empty_f64, empty_f64,     # bonds
            empty_i32, empty_i32, empty_i32,                 # angle indices
            empty_f64, empty_f64,                            # angle params
            empty_i32, empty_i32, empty_i32, empty_i32,     # dihedral indices
            empty_f64, empty_f64, empty_i32,                 # dihedral params
            pair_i_arr, pair_j_arr, sigma_arr, epsilon_arr,  # LJ pairs
            1.5,                                             # cutoff
        )

        start = time.perf_counter()
        for _ in range(n_iters):
            _rs.compute_all_forces(
                positions,
                empty_i32, empty_i32, empty_f64, empty_f64,
                empty_i32, empty_i32, empty_i32,
                empty_f64, empty_f64,
                empty_i32, empty_i32, empty_i32, empty_i32,
                empty_f64, empty_f64, empty_i32,
                pair_i_arr, pair_j_arr, sigma_arr, epsilon_arr,
                1.5,
            )
        elapsed = time.perf_counter() - start
        return (n_pairs * n_iters) / elapsed

    # No Rust backend — benchmark with vectorized NumPy LJ
    # Build all pair displacements at once
    sigma = 0.4
    epsilon = 1.0
    cutoff = 1.5

    # Warm up
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    start = time.perf_counter()
    for _ in range(n_iters):
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        # Upper triangle only (unique pairs)
        mask = np.triu(np.ones((n_beads, n_beads), dtype=bool), k=1)
        mask &= (dist > 1e-12) & (dist < cutoff)
        r = dist[mask]
        sig_r = sigma / np.maximum(r, 0.5 * sigma)
        sig_r6 = sig_r ** 6
        sig_r12 = sig_r6 ** 2
        _energy = np.sum(4.0 * epsilon * (sig_r12 - sig_r6))
    elapsed = time.perf_counter() - start

    # NumPy vectorized is faster than the Python loop in engine.py but
    # slower than Rust. Scale down by 0.5 to give a conservative estimate
    # that reflects the actual engine loop performance.
    return (n_pairs * n_iters) / elapsed * 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_hardware() -> HardwareProfile:
    """Auto-detect available compute resources."""
    cpu_cores = os.cpu_count() or 1
    cpu_name = _detect_cpu_name()
    gpu_available, gpu_name = _detect_gpu()
    memory_gb = _detect_memory_gb()
    estimated_pps = _benchmark_pairs_per_second()

    return HardwareProfile(
        cpu_cores=cpu_cores,
        cpu_name=cpu_name,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        memory_gb=round(memory_gb, 2),
        estimated_pairs_per_second=estimated_pps,
    )


def estimate_max_beads(
    profile: HardwareProfile,
    target_steps_per_second: float,
    dt: float = 0.01,
) -> int:
    """Given hardware and target step rate, estimate max beads.

    Non-bonded pairs scale as n*(n-1)/2.  Each step requires one full
    force evaluation over all pairs, so:

        pairs_per_step = n*(n-1)/2
        steps_per_second = pairs_per_second / pairs_per_step
        target_sps = pairs_per_second / (n*(n-1)/2)

    Solving for n:
        n*(n-1) = 2 * pairs_per_second / target_sps
        n ≈ sqrt(2 * pairs_per_second / target_sps)

    The dt parameter is accepted for interface consistency but does not
    affect the pair-throughput estimate (force evaluation dominates).
    """
    if target_steps_per_second <= 0:
        raise ValueError("target_steps_per_second must be positive")
    n = math.sqrt(2.0 * profile.estimated_pairs_per_second / target_steps_per_second)
    return max(int(n), 1)


def estimate_step_rate(profile: HardwareProfile, n_beads: int) -> float:
    """Estimate steps per second for a given bead count.

    Each step evaluates n*(n-1)/2 non-bonded pairs, so:
        steps/s = pairs_per_second / (n*(n-1)/2)
    """
    if n_beads < 2:
        return profile.estimated_pairs_per_second  # trivial system
    pairs_per_step = n_beads * (n_beads - 1) / 2.0
    return profile.estimated_pairs_per_second / pairs_per_step
