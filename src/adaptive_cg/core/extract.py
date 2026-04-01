"""Extract CG distributions from AA trajectories.

Given an AA trajectory and a CG mapping, this module:
1. Applies the mapping to each frame (atom positions → bead COM positions)
2. Classifies each bead by chemical composition
3. Detects bonded topology between beads
4. Extracts bond length, angle, and non-bonded distance distributions
   grouped by bead-class pair

These distributions feed into Boltzmann inversion for force field
parameterization.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Bead classification
# ---------------------------------------------------------------------------

# Atom-level chemical categories for bead typing
_POLAR_ELEMENTS = frozenset({"N", "O", "S"})
_CHARGED_RESNAMES = frozenset({
    "ARG", "LYS", "HIS", "HIP",  # positive
    "ASP", "GLU",                  # negative
})
_HYDROPHOBIC_RESNAMES = frozenset({
    "ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO",
})

# Backbone atom names (protein)
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


@dataclass
class BeadClass:
    """Chemical fingerprint of a CG bead."""
    n_atoms: int
    n_backbone: int
    n_sidechain: int
    n_polar: int
    n_carbon: int
    dominant_polarity: str  # "polar", "hydrophobic", "charged", "mixed"
    mol_type: str           # "protein", "nucleic_acid", "small_molecule"

    @property
    def label(self) -> str:
        """Human-readable bead class label."""
        return f"{self.mol_type}_{self.dominant_polarity}_{self.n_atoms}atoms"

    @property
    def key(self) -> str:
        """Hashable key for grouping similar beads.

        Groups by polarity and size bin (1-3, 4-6, 7-10, 11+) to avoid
        too many unique classes.
        """
        if self.n_atoms <= 3:
            size = "S"
        elif self.n_atoms <= 6:
            size = "M"
        elif self.n_atoms <= 10:
            size = "L"
        else:
            size = "XL"
        return f"{self.mol_type}_{self.dominant_polarity}_{size}"

    def to_dict(self) -> dict:
        return {
            "n_atoms": self.n_atoms,
            "n_backbone": self.n_backbone,
            "n_sidechain": self.n_sidechain,
            "n_polar": self.n_polar,
            "n_carbon": self.n_carbon,
            "dominant_polarity": self.dominant_polarity,
            "mol_type": self.mol_type,
            "label": self.label,
            "key": self.key,
        }


def classify_bead(
    atom_indices: list[int],
    elements: list[str],
    atom_names: list[str],
    residue_names: list[str],
    mol_type: str,
) -> BeadClass:
    """Classify a bead by its chemical composition."""
    n = len(atom_indices)
    n_backbone = 0
    n_polar = 0
    n_carbon = 0
    n_charged_res = 0
    n_hydrophobic_res = 0

    for idx in atom_indices:
        elem = elements[idx].strip().upper()
        aname = atom_names[idx].strip()
        resname = residue_names[idx].strip()

        if elem == "C":
            n_carbon += 1
        if elem in _POLAR_ELEMENTS:
            n_polar += 1
        if aname in _BACKBONE_ATOMS:
            n_backbone += 1
        if resname in _CHARGED_RESNAMES:
            n_charged_res += 1
        if resname in _HYDROPHOBIC_RESNAMES:
            n_hydrophobic_res += 1

    n_sidechain = n - n_backbone

    # Determine dominant polarity
    if n_charged_res > n // 2:
        polarity = "charged"
    elif n_polar > n_carbon:
        polarity = "polar"
    elif n_hydrophobic_res > n // 2:
        polarity = "hydrophobic"
    else:
        polarity = "mixed"

    return BeadClass(
        n_atoms=n,
        n_backbone=n_backbone,
        n_sidechain=n_sidechain,
        n_polar=n_polar,
        n_carbon=n_carbon,
        dominant_polarity=polarity,
        mol_type=mol_type,
    )


# ---------------------------------------------------------------------------
# Bonded topology detection
# ---------------------------------------------------------------------------

def detect_bonds(
    mapping: list[list[int]],
    n_atoms: int,
) -> list[tuple[int, int]]:
    """Detect bonded bead pairs from sequential atom connectivity.

    Two beads are bonded if any atom in one bead is sequentially adjacent
    (index difference == 1) to any atom in the other bead. This captures
    the backbone connectivity.

    Returns list of (bead_i, bead_j) tuples with i < j.
    """
    # Build atom → bead lookup
    atom_to_bead = np.full(n_atoms, -1, dtype=int)
    for bead_idx, group in enumerate(mapping):
        for atom_idx in group:
            atom_to_bead[atom_idx] = bead_idx

    bonds = set()
    for atom_idx in range(n_atoms - 1):
        bi = atom_to_bead[atom_idx]
        bj = atom_to_bead[atom_idx + 1]
        if bi != bj and bi >= 0 and bj >= 0:
            bonds.add((min(bi, bj), max(bi, bj)))

    return sorted(bonds)


def detect_angles(
    bonds: list[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    """Detect angle triplets (i, j, k) from bonded pairs.

    An angle exists when bead j is bonded to both bead i and bead k.
    """
    from collections import defaultdict
    neighbors = defaultdict(set)
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)

    angles = set()
    for j, nbrs in neighbors.items():
        nbrs_sorted = sorted(nbrs)
        for idx_a in range(len(nbrs_sorted)):
            for idx_b in range(idx_a + 1, len(nbrs_sorted)):
                i, k = nbrs_sorted[idx_a], nbrs_sorted[idx_b]
                angles.add((i, j, k))

    return sorted(angles)


def detect_dihedrals(
    bonds: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    """Detect dihedral quadruplets (i, j, k, l) from bonded pairs.

    A dihedral exists for each consecutive chain i-j-k-l where
    i-j, j-k, and k-l are all bonded.
    """
    from collections import defaultdict
    neighbors = defaultdict(set)
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)

    dihedrals = set()
    for j, k in bonds:
        for i in neighbors[j]:
            if i == k:
                continue
            for l in neighbors[k]:
                if l == j or l == i:
                    continue
                # Canonical ordering: ensure i < l to avoid duplicates
                if i < l:
                    dihedrals.add((i, j, k, l))
                else:
                    dihedrals.add((l, k, j, i))

    return sorted(dihedrals)


def compute_dihedral_angle(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
) -> float:
    """Compute dihedral angle between planes (p1,p2,p3) and (p2,p3,p4).

    Returns angle in radians, range [-pi, pi].
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-12 or n2_norm < 1e-12:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    cos_phi = np.clip(np.dot(n1, n2), -1.0, 1.0)
    # Sign from cross product
    sign = np.sign(np.dot(n1, b3))
    if sign == 0:
        sign = 1.0

    return float(sign * np.arccos(cos_phi))


# ---------------------------------------------------------------------------
# Distribution extraction
# ---------------------------------------------------------------------------

def compute_bead_positions(
    mapping: list[list[int]],
    positions: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Compute center-of-mass bead positions for one frame."""
    n_beads = len(mapping)
    bead_pos = np.empty((n_beads, 3))
    for i, group in enumerate(mapping):
        grp = np.asarray(group)
        m = masses[grp]
        bead_pos[i] = (m[:, None] * positions[grp]).sum(axis=0) / m.sum()
    return bead_pos


@dataclass
class ExtractionResult:
    """Distributions extracted from an AA trajectory."""
    molecule: str
    n_frames: int
    n_beads: int
    n_atoms: int

    # Bead classification
    bead_classes: list[dict]           # per-bead class info
    bead_class_keys: list[str]         # per-bead class key

    # Bond distributions: keyed by "classA--classB"
    bond_distributions: dict[str, list[float]]

    # Angle distributions: keyed by "classA--classB--classC"
    angle_distributions: dict[str, list[float]]

    # Dihedral distributions: keyed by "classA--classB--classC--classD"
    dihedral_distributions: dict[str, list[float]]

    # Non-bonded RDF samples: keyed by "classA--classB"
    nonbonded_distances: dict[str, list[float]]

    # Topology
    bonds: list[tuple[int, int]]
    angles: list[tuple[int, int, int]]
    dihedrals: list[tuple[int, int, int, int]]

    def save(self, output_dir: Path):
        """Save extraction results to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "molecule": self.molecule,
            "n_frames": int(self.n_frames),
            "n_beads": int(self.n_beads),
            "n_atoms": int(self.n_atoms),
            "bead_classes": self.bead_classes,
            "bead_class_keys": self.bead_class_keys,
            "bonds": [[int(x) for x in b] for b in self.bonds],
            "angles": [[int(x) for x in a] for a in self.angles],
            "dihedrals": [[int(x) for x in d] for d in self.dihedrals],
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save distributions as numpy arrays
        for name, dist in self.bond_distributions.items():
            np.save(output_dir / f"bond_{name}.npy", np.array(dist))

        for name, dist in self.angle_distributions.items():
            np.save(output_dir / f"angle_{name}.npy", np.array(dist))

        for name, dist in self.dihedral_distributions.items():
            np.save(output_dir / f"dihedral_{name}.npy", np.array(dist))

        for name, dist in self.nonbonded_distances.items():
            np.save(output_dir / f"nonbond_{name}.npy", np.array(dist))

    @staticmethod
    def load(input_dir: Path) -> "ExtractionResult":
        """Load extraction results from disk."""
        with open(input_dir / "meta.json") as f:
            meta = json.load(f)

        bond_dists = {}
        for p in input_dir.glob("bond_*.npy"):
            key = p.stem[5:]  # strip "bond_"
            bond_dists[key] = np.load(p).tolist()

        angle_dists = {}
        for p in input_dir.glob("angle_*.npy"):
            key = p.stem[6:]  # strip "angle_"
            angle_dists[key] = np.load(p).tolist()

        dihedral_dists = {}
        for p in input_dir.glob("dihedral_*.npy"):
            key = p.stem[9:]  # strip "dihedral_"
            dihedral_dists[key] = np.load(p).tolist()

        nonbond_dists = {}
        for p in input_dir.glob("nonbond_*.npy"):
            key = p.stem[8:]  # strip "nonbond_"
            nonbond_dists[key] = np.load(p).tolist()

        return ExtractionResult(
            molecule=meta["molecule"],
            n_frames=meta["n_frames"],
            n_beads=meta["n_beads"],
            n_atoms=meta["n_atoms"],
            bead_classes=meta["bead_classes"],
            bead_class_keys=meta["bead_class_keys"],
            bond_distributions=bond_dists,
            angle_distributions=angle_dists,
            dihedral_distributions=dihedral_dists,
            nonbonded_distances=nonbond_dists,
            bonds=[tuple(b) for b in meta["bonds"]],
            angles=[tuple(a) for a in meta["angles"]],
            dihedrals=[tuple(d) for d in meta.get("dihedrals", [])],
        )


def extract_distributions(
    trajectory_dir: Path,
    pdb_path: Path,
    n_beads: int | None = None,
    ratio: int = 4,
    nonbonded_cutoff: float = 2.0,
    max_nonbonded_pairs: int = 50,
    verbose: bool = True,
) -> ExtractionResult:
    """Extract CG distributions from an AA trajectory.

    Parameters
    ----------
    trajectory_dir : Path
        Directory containing *_solvated.pdb, *_traj.dcd, *_forces.npy.
    pdb_path : Path
        Original PDB file (for atom classification, heavy atoms only).
    n_beads : int or None
        Number of CG beads. If None, uses n_atoms // ratio.
    ratio : int
        Default atom-to-bead ratio if n_beads is not specified.
    nonbonded_cutoff : float
        Max distance (nm) for sampling non-bonded pairs.
    max_nonbonded_pairs : int
        Cap on non-bonded pair types to sample (avoid combinatorial explosion).
    verbose : bool
        Print progress.

    Returns
    -------
    ExtractionResult
    """
    import MDAnalysis as mda
    from adaptive_cg.core.molecule import (
        WATER_RESNAMES, ION_RESNAMES, _detect_mol_type,
    )
    from adaptive_cg.core.strategies import kmeans_mapping

    # Load heavy atoms directly — avoids DSSP dependency.
    # We only need positions, masses, elements, atom/residue names for
    # k-means mapping and bead classification. No region labels needed.
    pdb_id = pdb_path.stem.upper()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u_pdb = mda.Universe(str(pdb_path))

    exclude_list = " ".join(sorted(WATER_RESNAMES | ION_RESNAMES))
    sel_string = f"not type H and not resname {exclude_list}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ag = u_pdb.select_atoms(sel_string)

    # First chain only
    chains = sorted(set(ag.segids))
    if len(chains) > 1:
        ag = ag.select_atoms(f"segid {chains[0]}")

    positions = ag.positions / 10.0  # Angstroms → nm
    masses = ag.masses.copy()

    # Filter zero-mass atoms
    nonzero = masses > 0
    if not nonzero.all():
        positions = positions[nonzero]
        masses = masses[nonzero]
        ag = ag[nonzero]

    elements = [a.element.strip() if a.element else a.name.strip()[0] for a in ag]
    atom_names = [a.name.strip() for a in ag]
    residue_names = [a.resname.strip() for a in ag]
    n_atoms = len(ag)
    mol_type = _detect_mol_type(residue_names)

    if verbose:
        print(f"Loaded molecule: {pdb_id}, {n_atoms} heavy atoms")

    # Determine bead count
    if n_beads is None:
        n_beads = max(2, n_atoms // ratio)
    if verbose:
        print(f"Target beads: {n_beads}")

    # Generate mapping from static structure (k-means)
    mapping = kmeans_mapping(positions, masses, n_beads)
    if verbose:
        sizes = [len(g) for g in mapping]
        print(f"Mapping: {len(mapping)} beads, "
              f"sizes {min(sizes)}-{max(sizes)} atoms/bead")

    # Classify beads
    bead_classes_list = []
    bead_keys = []
    for group in mapping:
        bc = classify_bead(group, elements, atom_names, residue_names, mol_type)
        bead_classes_list.append(bc.to_dict())
        bead_keys.append(bc.key)

    if verbose:
        unique_keys = sorted(set(bead_keys))
        print(f"Bead classes: {len(unique_keys)} unique types")
        for k in unique_keys:
            count = bead_keys.count(k)
            print(f"  {k}: {count} beads")

    # Detect topology
    bonds = detect_bonds(mapping, n_atoms)
    angles = detect_angles(bonds)
    dihedrals = detect_dihedrals(bonds)
    if verbose:
        print(f"Topology: {len(bonds)} bonds, {len(angles)} angles, "
              f"{len(dihedrals)} dihedrals")

    # Load trajectory
    traj_dcd = list(trajectory_dir.glob("*_traj.dcd"))
    topo_pdb = list(trajectory_dir.glob("*_solvated.pdb"))
    if not traj_dcd or not topo_pdb:
        raise FileNotFoundError(
            f"Trajectory files not found in {trajectory_dir}. "
            "Run `acg simulate` first."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(str(topo_pdb[0]), str(traj_dcd[0]))

    n_frames = len(u.trajectory)
    if verbose:
        print(f"Trajectory: {n_frames} frames, {u.atoms.n_atoms} atoms (with solvent)")

    # We need to map solvated-system atom indices to our heavy-atom indices.
    # The solvated PDB has hydrogens + water + ions that our mapping doesn't cover.
    # Select the same heavy atoms from the solvated topology.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heavy_sel = u.select_atoms(sel_string)

    # First chain
    heavy_chains = sorted(set(heavy_sel.segids))
    if len(heavy_chains) > 1:
        heavy_sel = heavy_sel.select_atoms(f"segid {heavy_chains[0]}")

    if len(heavy_sel) != n_atoms:
        # Solvated system may have slightly different atom count due to
        # hydrogen addition changing residue topology. Use closest match.
        if verbose:
            print(f"  Warning: heavy atom count mismatch "
                  f"({len(heavy_sel)} vs {n_atoms}). "
                  f"Using min({len(heavy_sel)}, {n_atoms}) atoms.")
        n_use = min(len(heavy_sel), n_atoms)
        # Trim mapping to fit
        trimmed_mapping = []
        for group in mapping:
            trimmed = [i for i in group if i < n_use]
            if trimmed:
                trimmed_mapping.append(trimmed)
        mapping = trimmed_mapping

    heavy_indices = heavy_sel.indices  # indices into the solvated universe

    # --- Extract distributions across frames ---
    bond_dists: dict[str, list[float]] = {}
    angle_dists: dict[str, list[float]] = {}
    dihedral_dists: dict[str, list[float]] = {}
    nonbond_dists: dict[str, list[float]] = {}

    # Prepare bond class keys
    bond_class_keys = []
    for i, j in bonds:
        k = "--".join(sorted([bead_keys[i], bead_keys[j]]))
        bond_class_keys.append(k)
        if k not in bond_dists:
            bond_dists[k] = []

    # Prepare angle class keys
    angle_class_keys = []
    for i, j, k in angles:
        ak = "--".join([bead_keys[i], bead_keys[j], bead_keys[k]])
        angle_class_keys.append(ak)
        if ak not in angle_dists:
            angle_dists[ak] = []

    # Prepare dihedral class keys
    dihedral_class_keys = []
    for i, j, k, l in dihedrals:
        dk = "--".join([bead_keys[i], bead_keys[j], bead_keys[k], bead_keys[l]])
        dihedral_class_keys.append(dk)
        if dk not in dihedral_dists:
            dihedral_dists[dk] = []

    # Prepare non-bonded pairs (sample subset to avoid O(n²) explosion)
    bonded_set = set(bonds) | {(j, i) for i, j in bonds}
    nonbond_pairs = []
    for i in range(len(mapping)):
        for j in range(i + 1, len(mapping)):
            if (i, j) not in bonded_set:
                nonbond_pairs.append((i, j))
    # Subsample if too many
    if len(nonbond_pairs) > max_nonbonded_pairs:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(nonbond_pairs), max_nonbonded_pairs, replace=False)
        nonbond_pairs = [nonbond_pairs[i] for i in sorted(indices)]

    nonbond_class_keys = []
    for i, j in nonbond_pairs:
        k = "--".join(sorted([bead_keys[i], bead_keys[j]]))
        nonbond_class_keys.append(k)
        if k not in nonbond_dists:
            nonbond_dists[k] = []

    # Process each frame
    masses = masses
    for frame_idx, ts in enumerate(u.trajectory):
        # Get heavy atom positions in nm
        all_pos = ts.positions / 10.0  # Angstroms → nm
        heavy_pos = all_pos[heavy_indices]

        # Trim to match mapping
        if len(heavy_pos) > n_atoms:
            heavy_pos = heavy_pos[:n_atoms]

        # Compute bead COM positions
        bead_pos = compute_bead_positions(mapping, heavy_pos, masses)

        # Bond lengths
        for bond_idx, (i, j) in enumerate(bonds):
            dist = np.linalg.norm(bead_pos[i] - bead_pos[j])
            bond_dists[bond_class_keys[bond_idx]].append(float(dist))

        # Angles
        for angle_idx, (i, j, k) in enumerate(angles):
            v1 = bead_pos[i] - bead_pos[j]
            v2 = bead_pos[k] - bead_pos[j]
            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12
            )
            angle = float(np.arccos(np.clip(cos_angle, -1, 1)))
            angle_dists[angle_class_keys[angle_idx]].append(angle)

        # Dihedrals
        for dih_idx, (i, j, k, l) in enumerate(dihedrals):
            phi = compute_dihedral_angle(
                bead_pos[i], bead_pos[j], bead_pos[k], bead_pos[l],
            )
            dihedral_dists[dihedral_class_keys[dih_idx]].append(phi)

        # Non-bonded distances
        for nb_idx, (i, j) in enumerate(nonbond_pairs):
            dist = np.linalg.norm(bead_pos[i] - bead_pos[j])
            if dist < nonbonded_cutoff:
                nonbond_dists[nonbond_class_keys[nb_idx]].append(float(dist))

        if verbose and (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{n_frames} frames")

    if verbose:
        print(f"  Done. Bond types: {len(bond_dists)}, "
              f"Angle types: {len(angle_dists)}, "
              f"Dihedral types: {len(dihedral_dists)}, "
              f"Non-bonded types: {len(nonbond_dists)}")

    return ExtractionResult(
        molecule=pdb_id,
        n_frames=n_frames,
        n_beads=len(mapping),
        n_atoms=n_atoms,
        bead_classes=bead_classes_list,
        bead_class_keys=bead_keys,
        bond_distributions=bond_dists,
        angle_distributions=angle_dists,
        dihedral_distributions=dihedral_dists,
        nonbonded_distances=nonbond_dists,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
    )
