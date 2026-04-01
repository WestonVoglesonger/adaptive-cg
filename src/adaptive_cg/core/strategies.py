"""CG mapping optimization strategies.

Four approaches for generating coarse-grained mappings from atom
positions and masses:

- kmeans_mapping: K-means clustering with a connectivity penalty
- spectral_mapping: Spectral clustering on a molecular contact graph
- hierarchical_mapping: Agglomerative clustering (Ward linkage)
- annealing_mapping: Simulated annealing over sequential mappings
"""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist

from adaptive_cg.core.mapping import eval_mapping, generate_uniform_mapping

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Guarded sklearn imports
# ---------------------------------------------------------------------------

try:
    from sklearn.cluster import KMeans, SpectralClustering
except ImportError:  # pragma: no cover
    KMeans = None  # type: ignore[assignment,misc]
    SpectralClustering = None  # type: ignore[assignment,misc]


def _require_sklearn(name: str) -> None:
    """Raise a clear error if sklearn is missing."""
    if KMeans is None:
        raise ImportError(
            f"{name} requires scikit-learn.  "
            "Install it with:  pip install scikit-learn"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sort_clusters_by_mean_index(labels: np.ndarray, n_clusters: int) -> list[list[int]]:
    """Convert cluster labels to a mapping sorted by mean atom index.

    Parameters
    ----------
    labels : np.ndarray, shape (n_atoms,)
        Cluster label for each atom.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    list[list[int]]
        Sorted mapping (list of lists of atom indices).
    """
    clusters: dict[int, list[int]] = {k: [] for k in range(n_clusters)}
    for atom_idx, label in enumerate(labels):
        clusters[int(label)].append(atom_idx)

    # Sort clusters by the mean atom index within each cluster.
    sorted_groups = sorted(clusters.values(), key=lambda g: np.mean(g))
    return [sorted(g) for g in sorted_groups]


# ---------------------------------------------------------------------------
# 1. K-means mapping
# ---------------------------------------------------------------------------

def _merge_small_clusters(
    mapping: list[list[int]],
    positions: np.ndarray,
    min_atoms: int,
) -> list[list[int]]:
    """Merge clusters smaller than min_atoms into their nearest neighbor.

    Small beads (2-3 atoms) cause LJ explosions because their sigma is
    too small to provide adequate repulsion. Merging them into adjacent
    clusters eliminates the problem at the source.
    """
    if min_atoms <= 1:
        return mapping

    result = [list(g) for g in mapping]
    changed = True
    while changed:
        changed = False
        # Compute centroids
        centroids = np.array([positions[g].mean(axis=0) for g in result])
        i = 0
        while i < len(result):
            if len(result[i]) < min_atoms and len(result) > 1:
                # Find nearest neighbor cluster
                dists = np.array([
                    np.linalg.norm(centroids[i] - centroids[j])
                    if j != i else np.inf
                    for j in range(len(result))
                ])
                nearest = int(np.argmin(dists))
                # Merge into nearest
                result[nearest].extend(result[i])
                result[nearest].sort()
                result.pop(i)
                centroids = np.array([positions[g].mean(axis=0) for g in result])
                changed = True
            else:
                i += 1

    return result


def kmeans_mapping(
    positions: np.ndarray,
    masses: np.ndarray,
    n_beads: int,
    connectivity_weight: float = 0.5,
    min_atoms: int = 4,
) -> list[list[int]]:
    """K-means clustering on atom positions with a connectivity penalty.

    Atoms are embedded in a 4-D feature space
    ``[x, y, z, seq_index * connectivity_weight]`` where
    ``seq_index = atom_index / n_atoms``.  The extra dimension encourages
    sequentially nearby atoms to land in the same cluster.

    Clusters with fewer than min_atoms atoms are merged into their
    nearest neighbor to prevent LJ instabilities from tiny beads.

    Parameters
    ----------
    positions : np.ndarray, shape (n_atoms, 3)
        Atom positions in nm.
    masses : np.ndarray, shape (n_atoms,)
        Atom masses (unused but kept for consistent API).
    n_beads : int
        Number of CG beads (clusters).
    connectivity_weight : float
        Scaling factor for the sequential-index feature.
    min_atoms : int
        Minimum atoms per bead. Smaller clusters get merged (default: 4).

    Returns
    -------
    list[list[int]]
    """
    _require_sklearn("kmeans_mapping")

    n_atoms = positions.shape[0]
    seq_index = np.arange(n_atoms, dtype=float) / n_atoms

    # Build 4-D feature matrix: [x, y, z, seq * weight]
    features = np.column_stack([positions, seq_index * connectivity_weight])

    km = KMeans(n_clusters=n_beads, n_init=10, random_state=42)
    labels = km.fit_predict(features)

    mapping = _sort_clusters_by_mean_index(labels, n_beads)

    # Merge small clusters
    mapping = _merge_small_clusters(mapping, positions, min_atoms)

    return mapping


# ---------------------------------------------------------------------------
# 2. Spectral mapping
# ---------------------------------------------------------------------------

def spectral_mapping(
    positions: np.ndarray,
    masses: np.ndarray,
    n_beads: int,
    contact_cutoff: float = 0.5,
) -> list[list[int]]:
    """Spectral clustering on a molecular contact graph.

    Nodes are atoms.  An edge is added between atoms whose Euclidean
    distance is less than ``contact_cutoff`` (nm).  Sequential
    connectivity edges (atom *i* to atom *i+1*) are given higher weight
    to favour chemically bonded neighbours.

    Parameters
    ----------
    positions : np.ndarray, shape (n_atoms, 3)
        Atom positions in nm.
    masses : np.ndarray, shape (n_atoms,)
        Atom masses (unused but kept for consistent API).
    n_beads : int
        Number of CG beads (clusters).
    contact_cutoff : float
        Distance threshold in nm for contact edges.

    Returns
    -------
    list[list[int]]
    """
    _require_sklearn("spectral_mapping")

    n_atoms = positions.shape[0]
    dmat = cdist(positions, positions)

    # Build adjacency matrix: 1.0 where distance < cutoff, 0 on diagonal.
    adjacency = (dmat < contact_cutoff).astype(float)
    np.fill_diagonal(adjacency, 0.0)

    # Strengthen sequential (bonded) connectivity.
    seq_weight = 5.0
    for i in range(n_atoms - 1):
        adjacency[i, i + 1] = seq_weight
        adjacency[i + 1, i] = seq_weight

    sc = SpectralClustering(
        n_clusters=n_beads,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    labels = sc.fit_predict(adjacency)

    return _sort_clusters_by_mean_index(labels, n_beads)


# ---------------------------------------------------------------------------
# 3. Hierarchical mapping
# ---------------------------------------------------------------------------

def hierarchical_mapping(
    positions: np.ndarray,
    masses: np.ndarray,
    n_beads: int,
    method: str = "ward",
) -> list[list[int]]:
    """Agglomerative clustering on atom positions.

    Uses ``scipy.cluster.hierarchy.linkage`` + ``fcluster`` to partition
    atoms into ``n_beads`` clusters.  The default ``method='ward'``
    minimises within-cluster variance.

    Parameters
    ----------
    positions : np.ndarray, shape (n_atoms, 3)
        Atom positions in nm.
    masses : np.ndarray, shape (n_atoms,)
        Atom masses (unused but kept for consistent API).
    n_beads : int
        Number of CG beads (clusters).
    method : str
        Linkage method passed to ``scipy.cluster.hierarchy.linkage``.

    Returns
    -------
    list[list[int]]
    """
    Z = linkage(positions, method=method)
    # fcluster labels start at 1; convert to 0-based.
    labels = fcluster(Z, t=n_beads, criterion="maxclust") - 1

    return _sort_clusters_by_mean_index(labels, int(labels.max()) + 1)


# ---------------------------------------------------------------------------
# 4. Simulated-annealing mapping
# ---------------------------------------------------------------------------

def annealing_mapping(
    positions: np.ndarray,
    masses: np.ndarray,
    n_beads: int,
    n_iter: int = 5000,
    temp_start: float = 1.0,
    temp_end: float = 0.01,
) -> list[list[int]]:
    """Simulated annealing over sequential CG mappings.

    Starts from a uniform sequential mapping (via
    ``generate_uniform_mapping``).  Each iteration randomly swaps a pair
    of atoms across two adjacent beads.  Worse moves are accepted with
    Boltzmann probability ``exp(-delta / temp)`` under a linear
    temperature schedule.

    Parameters
    ----------
    positions : np.ndarray, shape (n_atoms, 3)
        Atom positions in nm.
    masses : np.ndarray, shape (n_atoms,)
        Atom masses.
    n_beads : int
        Number of CG beads.
    n_iter : int
        Number of annealing iterations.
    temp_start : float
        Initial temperature.
    temp_end : float
        Final temperature.

    Returns
    -------
    list[list[int]]
    """
    n_atoms = positions.shape[0]
    ratio = max(1, n_atoms // n_beads)

    # Initial mapping: uniform sequential.
    current = generate_uniform_mapping(n_atoms, ratio)

    # Pad or trim to exactly n_beads groups.
    while len(current) < n_beads:
        # Split the largest group.
        largest_idx = max(range(len(current)), key=lambda i: len(current[i]))
        grp = current[largest_idx]
        mid = len(grp) // 2
        current[largest_idx] = grp[:mid]
        current.insert(largest_idx + 1, grp[mid:])
    while len(current) > n_beads:
        # Merge the two smallest adjacent groups.
        min_idx = min(
            range(len(current) - 1),
            key=lambda i: len(current[i]) + len(current[i + 1]),
        )
        current[min_idx] = current[min_idx] + current[min_idx + 1]
        del current[min_idx + 1]

    # Pre-compute all-atom distance matrix for fast eval_mapping calls.
    aa_dmat = cdist(positions, positions)

    current_score = eval_mapping(current, positions, masses, aa_dmat=aa_dmat)["rmse"]
    best_mapping = [list(g) for g in current]
    best_score = current_score

    rng = random.Random(42)

    for iteration in range(n_iter):
        # Linear temperature annealing.
        temp = temp_start + (temp_end - temp_start) * (iteration / max(n_iter - 1, 1))

        # Pick a random pair of adjacent beads.
        bead_i = rng.randint(0, n_beads - 2)
        bead_j = bead_i + 1

        # Both groups must have at least 2 atoms to allow a swap.
        if len(current[bead_i]) < 2 or len(current[bead_j]) < 2:
            continue

        # Randomly select one atom from each group and swap them.
        idx_a = rng.randrange(len(current[bead_i]))
        idx_b = rng.randrange(len(current[bead_j]))

        atom_a = current[bead_i][idx_a]
        atom_b = current[bead_j][idx_b]

        # Apply the swap.
        current[bead_i][idx_a] = atom_b
        current[bead_j][idx_b] = atom_a

        new_score = eval_mapping(current, positions, masses, aa_dmat=aa_dmat)["rmse"]
        delta = new_score - current_score

        if delta < 0:
            # Improvement: always accept.
            current_score = new_score
        else:
            # Worse: accept with Boltzmann probability.
            acceptance = math.exp(-delta / temp) if temp > 1e-12 else 0.0
            if rng.random() < acceptance:
                current_score = new_score
            else:
                # Revert the swap.
                current[bead_i][idx_a] = atom_a
                current[bead_j][idx_b] = atom_b

        # Track the best mapping seen.
        if current_score < best_score:
            best_score = current_score
            best_mapping = [list(g) for g in current]

    # Sort atoms within each bead for consistency.
    return [sorted(g) for g in best_mapping]
