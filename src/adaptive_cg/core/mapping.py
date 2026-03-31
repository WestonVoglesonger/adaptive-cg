"""CG mapping evaluation and generation.

A mapping is a list of lists of atom indices, where each inner list
defines the atoms grouped into a single coarse-grained bead.
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    pass

# Threshold (number of atoms) above which we switch to chunked distance
# computation to avoid allocating an O(N^2) all-atom distance matrix.
_CHUNK_THRESHOLD = 1000


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_mapping(
    mapping: list[list[int]],
    positions: np.ndarray,
    masses: np.ndarray,
    aa_dmat: np.ndarray | None = None,
) -> dict:
    """Evaluate how well a CG mapping preserves pairwise bead distances.

    Parameters
    ----------
    mapping : list[list[int]]
        Each inner list contains atom indices for one bead.
    positions : np.ndarray, shape (n_atoms, 3)
        Atom positions in nm.
    masses : np.ndarray, shape (n_atoms,)
        Atom masses.
    aa_dmat : np.ndarray or None
        Pre-computed all-atom distance matrix. If provided, avoids
        recomputing it (major speedup when calling eval_mapping many
        times for the same molecule, e.g. during grid search).

    Returns
    -------
    dict with keys: rmse, mae, mre, n_beads, cg_dmat, ref_dmat.
    """
    n_beads = len(mapping)

    # --- CG bead positions (center of mass) ---
    bead_pos = np.empty((n_beads, 3))
    for i, group in enumerate(mapping):
        grp = np.asarray(group)
        m = masses[grp]
        bead_pos[i] = (m[:, None] * positions[grp]).sum(axis=0) / m.sum()

    # --- CG distance matrix ---
    cg_dmat = cdist(bead_pos, bead_pos)

    # --- Reference distance matrix (mean atom-atom distance per bead pair) ---
    n_atoms = positions.shape[0]

    if aa_dmat is not None:
        ref_dmat = _ref_dmat_from_precomputed(mapping, aa_dmat, n_beads)
    elif n_atoms <= _CHUNK_THRESHOLD:
        ref_dmat = _ref_dmat_full(mapping, positions, n_beads)
    else:
        ref_dmat = _ref_dmat_chunked(mapping, positions, n_beads)

    # --- Compare upper triangle only ---
    triu_idx = np.triu_indices(n_beads, k=1)
    cg_upper = cg_dmat[triu_idx]
    ref_upper = ref_dmat[triu_idx]

    if len(cg_upper) == 0:
        # Single bead or empty mapping: no pairs to compare.
        rmse = mae = mre = 0.0
    else:
        diff = cg_upper - ref_upper
        abs_diff = np.abs(diff)

        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(abs_diff))

        # Mean relative error: avoid division by zero for very close beads.
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(ref_upper > 1e-12, abs_diff / ref_upper, 0.0)
        mre = float(np.mean(rel))

    return {
        "rmse": rmse,
        "mae": mae,
        "mre": mre,
        "n_beads": n_beads,
        "cg_dmat": cg_dmat,
        "ref_dmat": ref_dmat,
    }


def _ref_dmat_from_precomputed(
    mapping: list[list[int]],
    aa_dmat: np.ndarray,
    n_beads: int,
) -> np.ndarray:
    """Compute reference distance matrix from a pre-computed AA distance matrix."""
    ref = np.zeros((n_beads, n_beads))
    for i in range(n_beads):
        gi = np.asarray(mapping[i])
        for j in range(i + 1, n_beads):
            gj = np.asarray(mapping[j])
            ref[i, j] = ref[j, i] = aa_dmat[np.ix_(gi, gj)].mean()
    return ref


def _ref_dmat_full(
    mapping: list[list[int]],
    positions: np.ndarray,
    n_beads: int,
) -> np.ndarray:
    """Compute reference distance matrix using the full atom-atom distance matrix.

    Fast for small molecules (n_atoms <= ~1000).
    """
    aa_dmat = cdist(positions, positions)
    ref = np.zeros((n_beads, n_beads))
    for i in range(n_beads):
        gi = np.asarray(mapping[i])
        for j in range(i + 1, n_beads):
            gj = np.asarray(mapping[j])
            block = aa_dmat[np.ix_(gi, gj)]
            ref[i, j] = ref[j, i] = block.mean()
    return ref


def _ref_dmat_chunked(
    mapping: list[list[int]],
    positions: np.ndarray,
    n_beads: int,
) -> np.ndarray:
    """Compute reference distance matrix without building the full AA dmat.

    For each bead pair (i, j), compute cdist only on the atoms in those
    two groups. Avoids O(N^2) memory for large molecules.
    """
    ref = np.zeros((n_beads, n_beads))
    for i in range(n_beads):
        pi = positions[mapping[i]]
        for j in range(i + 1, n_beads):
            pj = positions[mapping[j]]
            ref[i, j] = ref[j, i] = cdist(pi, pj).mean()
    return ref


# ---------------------------------------------------------------------------
# Mapping generators
# ---------------------------------------------------------------------------

def generate_uniform_mapping(n_atoms: int, ratio: int) -> list[list[int]]:
    """Create a sequential mapping with uniform bead size.

    Each bead gets ``ratio`` consecutive atoms; the last bead absorbs
    any remainder.

    Parameters
    ----------
    n_atoms : int
        Total number of atoms.
    ratio : int
        Target number of atoms per bead.

    Returns
    -------
    list[list[int]]
    """
    mapping: list[list[int]] = []
    for start in range(0, n_atoms, ratio):
        end = min(start + ratio, n_atoms)
        mapping.append(list(range(start, end)))

    # Merge trailing fragment into the last full bead if the remainder
    # would create a group smaller than 1 atom (safety; always at least 1).
    # The spec says "last bead gets remaining atoms", which the loop above
    # already handles by clamping `end` to n_atoms.
    return mapping


def generate_variable_mapping(
    n_atoms: int,
    region_labels: np.ndarray,
    region_ratios: dict[int, int],
) -> list[list[int]]:
    """Create a mapping with per-region bead sizes.

    Atoms are processed in order.  Contiguous runs of the same region
    label are grouped together, and within each run the bead size is
    determined by ``region_ratios[label]``.

    Parameters
    ----------
    n_atoms : int
        Total number of atoms.
    region_labels : np.ndarray, shape (n_atoms,)
        Integer region label for each atom.
    region_ratios : dict[int, int]
        Mapping from region label to atom-to-bead ratio for that region.

    Returns
    -------
    list[list[int]]
    """
    mapping: list[list[int]] = []

    # Identify contiguous blocks of same region label.
    blocks: list[tuple[int, int, int]] = []  # (label, start, end)
    if n_atoms == 0:
        return mapping

    labels = np.asarray(region_labels, dtype=int)
    current_label = int(labels[0])
    block_start = 0

    for idx in range(1, n_atoms):
        if int(labels[idx]) != current_label:
            blocks.append((current_label, block_start, idx))
            current_label = int(labels[idx])
            block_start = idx
    blocks.append((current_label, block_start, n_atoms))

    # Within each block, create beads of the specified ratio.
    for label, start, end in blocks:
        ratio = region_ratios[label]
        for bead_start in range(start, end, ratio):
            bead_end = min(bead_start + ratio, end)
            mapping.append(list(range(bead_start, bead_end)))

    return mapping


# ---------------------------------------------------------------------------
# Search / baselines
# ---------------------------------------------------------------------------

def grid_search_variable(
    positions: np.ndarray,
    masses: np.ndarray,
    region_labels: np.ndarray,
    region_names: list[str],
    ratio_range: tuple[int, int] = (2, 15),
    target_beads: int | None = None,
    tolerance: int = 3,
) -> dict:
    """Grid search over per-region ratios to find the best variable mapping.

    Parameters
    ----------
    positions : np.ndarray, shape (n_atoms, 3)
    masses : np.ndarray, shape (n_atoms,)
    region_labels : np.ndarray, shape (n_atoms,)
        Integer region label per atom.
    region_names : list[str]
        Human-readable name for each unique region (ordered by label value).
    ratio_range : tuple[int, int]
        Inclusive (lo, hi) range of ratios to try per region.
    target_beads : int | None
        If set, discard mappings whose bead count is outside
        ``target_beads +/- tolerance``.
    tolerance : int
        Allowed deviation from target_beads.

    Returns
    -------
    dict with keys: best_mapping, best_rmse, best_ratios, n_beads, all_results.
    """
    n_atoms = positions.shape[0]
    unique_labels = np.unique(region_labels).tolist()
    lo, hi = ratio_range
    ratio_values = list(range(lo, hi + 1))

    # Build all ratio combinations (one per unique region).
    combos = list(itertools.product(ratio_values, repeat=len(unique_labels)))

    # Pre-compute atom-atom distance matrix ONCE for the whole grid search.
    # This is the key optimisation: without it, every eval_mapping call
    # recomputes the N×N cdist, which causes OOM on M1 Macs for large sweeps.
    aa_dmat = cdist(positions, positions) if n_atoms <= _CHUNK_THRESHOLD else None

    all_results: list[dict] = []
    best_rmse = float("inf")
    best_result: dict | None = None

    for combo in combos:
        region_ratios = dict(zip(unique_labels, combo))
        m = generate_variable_mapping(n_atoms, region_labels, region_ratios)
        n_beads = len(m)

        # Filter by target bead count.
        if target_beads is not None:
            if abs(n_beads - target_beads) > tolerance:
                continue

        result = eval_mapping(m, positions, masses, aa_dmat=aa_dmat)

        # Augment with config info.
        ratio_dict = {
            name: region_ratios[label]
            for label, name in zip(unique_labels, region_names)
        }
        entry = {
            "ratios": ratio_dict,
            "rmse": result["rmse"],
            "mae": result["mae"],
            "mre": result["mre"],
            "n_beads": n_beads,
        }
        all_results.append(entry)

        if result["rmse"] < best_rmse:
            best_rmse = result["rmse"]
            best_result = {
                "best_mapping": m,
                "best_rmse": result["rmse"],
                "best_ratios": ratio_dict,
                "n_beads": n_beads,
            }

    if best_result is None:
        return {
            "best_mapping": [],
            "best_rmse": float("inf"),
            "best_ratios": {},
            "n_beads": 0,
            "all_results": all_results,
        }

    best_result["all_results"] = all_results
    return best_result


def eval_uniform_baselines(
    positions: np.ndarray,
    masses: np.ndarray,
    ratios: list[int],
) -> list[dict]:
    """Evaluate uniform CG mappings at several atom-to-bead ratios.

    Parameters
    ----------
    positions : np.ndarray, shape (n_atoms, 3)
    masses : np.ndarray, shape (n_atoms,)
    ratios : list[int]
        List of uniform ratios to evaluate.

    Returns
    -------
    list[dict]
        Each dict contains the eval_mapping results plus a ``ratio`` key.
    """
    n_atoms = positions.shape[0]
    aa_dmat = cdist(positions, positions) if n_atoms <= _CHUNK_THRESHOLD else None
    results = []
    for ratio in ratios:
        m = generate_uniform_mapping(n_atoms, ratio)
        res = eval_mapping(m, positions, masses, aa_dmat=aa_dmat)
        res["ratio"] = ratio
        results.append(res)
    return results
