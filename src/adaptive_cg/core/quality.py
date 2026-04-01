"""Quality metrics for CG simulation validation against AA reference."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class QualityMetrics:
    rg: float                       # radius of gyration (nm)
    rg_reference: float             # AA reference Rg (nm)
    rg_deviation: float             # |rg - rg_ref| / rg_ref
    contact_map_correlation: float  # correlation of CG vs AA contact maps
    rmsf_correlation: float         # per-region RMSF correlation with AA
    structural_quality: float       # combined score 0-1 (weighted average)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_rg(positions: np.ndarray) -> float:
    """Radius of gyration from bead positions (n, 3).

    Rg = sqrt(mean(|r_i - r_com|^2))
    """
    com = positions.mean(axis=0)
    displacements = positions - com
    return float(np.sqrt(np.mean(np.sum(displacements ** 2, axis=1))))


def compute_contact_map(positions: np.ndarray, cutoff: float = 0.8) -> np.ndarray:
    """Binary contact map: 1 if distance < cutoff. Returns (n, n) bool array."""
    dmat = cdist(positions, positions)
    return dmat < cutoff


def compute_rmsf(trajectory_frames: np.ndarray) -> np.ndarray:
    """Per-bead RMSF from trajectory frames (n_frames, n_beads, 3).

    RMSF_i = sqrt(mean_t(|r_i(t) - <r_i>|^2))
    """
    mean_positions = trajectory_frames.mean(axis=0)  # (n_beads, 3)
    deviations = trajectory_frames - mean_positions   # (n_frames, n_beads, 3)
    msd = np.mean(np.sum(deviations ** 2, axis=2), axis=0)  # (n_beads,)
    return np.sqrt(msd)


def compute_region_rmsf(rmsf: np.ndarray, n_regions: int) -> np.ndarray:
    """Average RMSF per sequential region.

    Splits beads into n_regions contiguous chunks and averages RMSF
    within each chunk. Final chunk absorbs any remainder beads.
    """
    n_beads = len(rmsf)
    if n_regions <= 0:
        raise ValueError("n_regions must be positive")
    if n_regions > n_beads:
        n_regions = n_beads

    chunk_size = n_beads // n_regions
    region_rmsf = np.empty(n_regions)
    for i in range(n_regions):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_regions - 1 else n_beads
        region_rmsf[i] = rmsf[start:end].mean()
    return region_rmsf


# ---------------------------------------------------------------------------
# Combined quality
# ---------------------------------------------------------------------------

def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flat arrays, clamped to [0, 1].

    Returns 0.0 if either array has zero variance (no information).
    """
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)

    if len(a_flat) != len(b_flat):
        # Truncate to shorter length (CG vs AA may differ in size)
        n = min(len(a_flat), len(b_flat))
        a_flat = a_flat[:n]
        b_flat = b_flat[:n]

    if len(a_flat) < 2:
        return 0.0

    a_std = a_flat.std()
    b_std = b_flat.std()
    if a_std == 0.0 or b_std == 0.0:
        return 0.0

    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    # Clamp: negative correlation is worse than no correlation for
    # structural similarity, so floor at 0.
    return float(max(0.0, corr))


def compute_quality(
    cg_positions: np.ndarray,
    cg_trajectory: np.ndarray | None,
    aa_positions: np.ndarray,
    aa_trajectory: np.ndarray | None,
    n_regions: int = 5,
) -> QualityMetrics:
    """Compute all quality metrics comparing CG to AA reference.

    Parameters
    ----------
    cg_positions : np.ndarray, shape (n_beads, 3)
        Current CG bead positions in nm.
    cg_trajectory : np.ndarray or None, shape (n_frames, n_beads, 3)
        CG trajectory frames. If None, RMSF correlation is set to 0.
    aa_positions : np.ndarray, shape (n_atoms, 3)
        AA atom positions from PDB in nm.
    aa_trajectory : np.ndarray or None, shape (n_frames, n_atoms, 3)
        AA trajectory frames. If None, RMSF correlation is set to 0.
    n_regions : int
        Number of sequential regions for RMSF comparison.

    Returns
    -------
    QualityMetrics
        Combined quality assessment. structural_quality is a weighted
        mean of (1 - rg_deviation), contact_map_correlation, and
        rmsf_correlation, clamped to [0, 1].
    """
    # --- Radius of gyration ---
    rg_cg = compute_rg(cg_positions)
    rg_aa = compute_rg(aa_positions)
    rg_deviation = abs(rg_cg - rg_aa) / rg_aa if rg_aa > 0 else 0.0

    # --- Contact map correlation ---
    # CG and AA have different numbers of particles, so we compare the
    # internal contact patterns of each (self-consistency of distances).
    cg_contacts = compute_contact_map(cg_positions)
    aa_contacts = compute_contact_map(aa_positions)
    contact_corr = _correlate(cg_contacts, aa_contacts)

    # --- RMSF correlation ---
    if cg_trajectory is not None and aa_trajectory is not None:
        cg_rmsf = compute_rmsf(cg_trajectory)
        aa_rmsf = compute_rmsf(aa_trajectory)
        cg_region = compute_region_rmsf(cg_rmsf, n_regions)
        aa_region = compute_region_rmsf(aa_rmsf, n_regions)
        rmsf_corr = _correlate(cg_region, aa_region)
    else:
        rmsf_corr = 0.0

    # --- Weighted combination ---
    # Weights: Rg fidelity 0.3, contact map 0.4, RMSF pattern 0.3
    w_rg, w_contact, w_rmsf = 0.3, 0.4, 0.3
    rg_score = max(0.0, 1.0 - rg_deviation)  # clamp negative

    if cg_trajectory is not None and aa_trajectory is not None:
        structural_quality = (
            w_rg * rg_score + w_contact * contact_corr + w_rmsf * rmsf_corr
        )
    else:
        # Without trajectory data, only Rg and contacts are available.
        # Renormalize weights to sum to 1.
        w_total = w_rg + w_contact
        structural_quality = (w_rg * rg_score + w_contact * contact_corr) / w_total

    structural_quality = float(np.clip(structural_quality, 0.0, 1.0))

    return QualityMetrics(
        rg=rg_cg,
        rg_reference=rg_aa,
        rg_deviation=rg_deviation,
        contact_map_correlation=contact_corr,
        rmsf_correlation=rmsf_corr,
        structural_quality=structural_quality,
    )


def meets_quality_floor(metrics: QualityMetrics, min_quality: float = 0.5) -> bool:
    """Check if simulation meets minimum quality threshold."""
    return metrics.structural_quality >= min_quality
