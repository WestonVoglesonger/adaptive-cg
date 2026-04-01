"""CG molecular dynamics engine.

Velocity Verlet integrator with Langevin thermostat, using harmonic
bond/angle potentials and Lennard-Jones non-bonded interactions loaded
from a parameterized force field (cg_forcefield.json).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# Boltzmann constant in kJ/(mol·K)
KB = 0.008314462618

# Try to import Rust backend
try:
    import cg_engine as _rs
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


# ---------------------------------------------------------------------------
# Force field loader
# ---------------------------------------------------------------------------

@dataclass
class BondParam:
    r0: float   # equilibrium distance (nm)
    k: float    # force constant (kJ/mol/nm²)


@dataclass
class AngleParam:
    theta0: float  # equilibrium angle (radians)
    k: float       # force constant (kJ/mol/rad²)


@dataclass
class LJParam:
    sigma: float    # nm
    epsilon: float  # kJ/mol


@dataclass
class DihedralParam:
    phi0: float     # phase angle (radians)
    k: float        # force constant (kJ/mol)
    n: int          # multiplicity


def load_forcefield(path: Path) -> dict:
    """Load force field parameters from JSON.

    Returns dict with keys: bonds, angles, dihedrals, nonbonded, temperature.
    Values are dicts keyed by class pair strings.
    """
    with open(path) as f:
        data = json.load(f)

    bonds = {}
    for key, v in data["bonds"].items():
        bonds[key] = BondParam(r0=v["x0"], k=v["k"])

    angles = {}
    for key, v in data["angles"].items():
        angles[key] = AngleParam(theta0=v["x0"], k=v["k"])

    dihedrals = {}
    for key, v in data.get("dihedrals", {}).items():
        dihedrals[key] = DihedralParam(phi0=v["phi0"], k=v["k"], n=v["n"])

    nonbonded = {}
    for key, v in data["nonbonded"].items():
        nonbonded[key] = LJParam(sigma=v["sigma"], epsilon=v["epsilon"])

    return {
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
        "nonbonded": nonbonded,
        "temperature": data["temperature"],
    }


# ---------------------------------------------------------------------------
# Force computation
# ---------------------------------------------------------------------------

def compute_bond_forces(
    positions: np.ndarray,
    bond_list: list[tuple[int, int]],
    bond_params: list[BondParam],
) -> tuple[np.ndarray, float]:
    """Compute harmonic bond forces and energy.

    U = k * (r - r0)²
    F_i = -dU/dr * r_hat  (toward j if stretched, away if compressed)
    """
    n = positions.shape[0]
    forces = np.zeros((n, 3))
    energy = 0.0

    for (i, j), param in zip(bond_list, bond_params):
        rij = positions[j] - positions[i]
        r = np.linalg.norm(rij)
        if r < 1e-12:
            continue
        r_hat = rij / r

        dr = r - param.r0
        # F = -dU/dr = -2k(r - r0), but our U = k(r-r0)² so dU/dr = 2k(r-r0)
        f_mag = -2.0 * param.k * dr
        f_vec = f_mag * r_hat

        forces[i] -= f_vec
        forces[j] += f_vec
        energy += param.k * dr * dr

    return forces, energy


def compute_angle_forces(
    positions: np.ndarray,
    angle_list: list[tuple[int, int, int]],
    angle_params: list[AngleParam],
) -> tuple[np.ndarray, float]:
    """Compute harmonic angle forces and energy.

    For angle i-j-k (j is the central bead):
    U = k * (theta - theta0)²
    """
    n = positions.shape[0]
    forces = np.zeros((n, 3))
    energy = 0.0

    for (i, j, k), param in zip(angle_list, angle_params):
        # Vectors from central bead j
        v1 = positions[i] - positions[j]
        v2 = positions[k] - positions[j]
        r1 = np.linalg.norm(v1)
        r2 = np.linalg.norm(v2)
        if r1 < 1e-12 or r2 < 1e-12:
            continue

        cos_theta = np.dot(v1, v2) / (r1 * r2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        sin_theta = np.sin(theta)
        if abs(sin_theta) < 1e-12:
            continue

        dtheta = theta - param.theta0
        energy += param.k * dtheta * dtheta

        # dU/dtheta = 2k(theta - theta0)
        coeff = -2.0 * param.k * dtheta / sin_theta

        # Forces on i and k (gradient of angle w.r.t. positions)
        # d(theta)/d(r_i) = (cos_theta * v1/r1 - v2/r2) / (r1 * sin_theta)
        f_i = coeff * (cos_theta * v1 / r1 - v2 / r2) / r1
        f_k = coeff * (cos_theta * v2 / r2 - v1 / r1) / r2
        f_j = -(f_i + f_k)

        forces[i] += f_i
        forces[j] += f_j
        forces[k] += f_k

    return forces, energy


def compute_nonbonded_forces(
    positions: np.ndarray,
    nb_pairs: list[tuple[int, int]],
    nb_params: list[LJParam],
    cutoff: float = 1.5,
) -> tuple[np.ndarray, float]:
    """Compute Lennard-Jones non-bonded forces and energy.

    U = 4*eps * [(sig/r)^12 - (sig/r)^6]
    F = 24*eps/r * [2*(sig/r)^12 - (sig/r)^6] * r_hat
    """
    n = positions.shape[0]
    forces = np.zeros((n, 3))
    energy = 0.0

    for (i, j), param in zip(nb_pairs, nb_params):
        rij = positions[j] - positions[i]
        r = np.linalg.norm(rij)
        if r < 1e-12 or r > cutoff:
            continue

        # Soft minimum distance to prevent catastrophic forces
        r = max(r, 0.5 * param.sigma)

        r_hat = rij / r
        sig_r = param.sigma / r
        sig_r6 = sig_r ** 6
        sig_r12 = sig_r6 ** 2

        # Energy
        energy += 4.0 * param.epsilon * (sig_r12 - sig_r6)

        # Force magnitude: F = 24*eps/r * (2*sig_r12 - sig_r6)
        f_mag = 24.0 * param.epsilon / r * (2.0 * sig_r12 - sig_r6)
        f_vec = f_mag * r_hat

        forces[i] -= f_vec
        forces[j] += f_vec

    return forces, energy


# ---------------------------------------------------------------------------
# System setup
# ---------------------------------------------------------------------------

@dataclass
class CGSystem:
    """A coarse-grained system ready for simulation."""
    n_beads: int
    positions: np.ndarray       # (n_beads, 3) nm
    velocities: np.ndarray      # (n_beads, 3) nm/ps
    masses: np.ndarray           # (n_beads,) daltons
    bead_class_keys: list[str]  # class key per bead

    # Topology with matched parameters
    bond_list: list[tuple[int, int]]
    bond_params: list[BondParam]
    angle_list: list[tuple[int, int, int]]
    angle_params: list[AngleParam]
    dihedral_list: list[tuple[int, int, int, int]]
    dihedral_params: list[DihedralParam]
    nb_pairs: list[tuple[int, int]]
    nb_params: list[LJParam]

    # Cached flat arrays for Rust backend (built lazily)
    _topo_arrays: dict | None = field(default=None, repr=False)

    def _get_topo_arrays(self) -> dict:
        """Build flat numpy arrays from topology lists (cached)."""
        if self._topo_arrays is not None:
            return self._topo_arrays
        bl = self.bond_list
        self._topo_arrays = {
            "bond_i": np.array([b[0] for b in bl], dtype=np.int64),
            "bond_j": np.array([b[1] for b in bl], dtype=np.int64),
            "bond_r0": np.array([p.r0 for p in self.bond_params]),
            "bond_k": np.array([p.k for p in self.bond_params]),
            "angle_i": np.array([a[0] for a in self.angle_list], dtype=np.int64),
            "angle_j": np.array([a[1] for a in self.angle_list], dtype=np.int64),
            "angle_k": np.array([a[2] for a in self.angle_list], dtype=np.int64),
            "angle_theta0": np.array([p.theta0 for p in self.angle_params]),
            "angle_k_param": np.array([p.k for p in self.angle_params]),
            "dih_i": np.array([d[0] for d in self.dihedral_list], dtype=np.int64),
            "dih_j": np.array([d[1] for d in self.dihedral_list], dtype=np.int64),
            "dih_k": np.array([d[2] for d in self.dihedral_list], dtype=np.int64),
            "dih_l": np.array([d[3] for d in self.dihedral_list], dtype=np.int64),
            "dih_phi0": np.array([p.phi0 for p in self.dihedral_params]),
            "dih_k_param": np.array([p.k for p in self.dihedral_params]),
            "dih_n": np.array([p.n for p in self.dihedral_params], dtype=np.int64),
            "pair_i": np.array([p[0] for p in self.nb_pairs], dtype=np.int64),
            "pair_j": np.array([p[1] for p in self.nb_pairs], dtype=np.int64),
            "pair_sigma": np.array([p.sigma for p in self.nb_params]),
            "pair_epsilon": np.array([p.epsilon for p in self.nb_params]),
        }
        return self._topo_arrays

    def compute_forces(self, cutoff: float = 1.5) -> tuple[np.ndarray, dict]:
        """Compute total forces and per-component energies."""
        if _HAS_RUST:
            t = self._get_topo_arrays()
            forces, e_bond, e_angle, e_dih, e_nb, e_total = _rs.compute_all_forces(
                self.positions, t["bond_i"], t["bond_j"], t["bond_r0"], t["bond_k"],
                t["angle_i"], t["angle_j"], t["angle_k"],
                t["angle_theta0"], t["angle_k_param"],
                t["dih_i"], t["dih_j"], t["dih_k"], t["dih_l"],
                t["dih_phi0"], t["dih_k_param"], t["dih_n"],
                t["pair_i"], t["pair_j"], t["pair_sigma"], t["pair_epsilon"],
                cutoff,
            )
            energies = {
                "bond": e_bond, "angle": e_angle, "dihedral": e_dih,
                "nonbonded": e_nb, "potential": e_total,
            }
            return np.asarray(forces), energies

        f_bond, e_bond = compute_bond_forces(
            self.positions, self.bond_list, self.bond_params
        )
        f_angle, e_angle = compute_angle_forces(
            self.positions, self.angle_list, self.angle_params
        )
        f_nb, e_nb = compute_nonbonded_forces(
            self.positions, self.nb_pairs, self.nb_params, cutoff
        )
        total_forces = f_bond + f_angle + f_nb
        energies = {
            "bond": e_bond, "angle": e_angle,
            "nonbonded": e_nb, "potential": e_bond + e_angle + e_nb,
        }
        return total_forces, energies

    def kinetic_energy(self) -> float:
        """Compute kinetic energy: sum(0.5 * m * v²)."""
        return 0.5 * np.sum(self.masses[:, None] * self.velocities ** 2)

    def temperature(self) -> float:
        """Instantaneous temperature from kinetic energy."""
        ke = self.kinetic_energy()
        n_dof = max(3 * self.n_beads - 3, 1)
        return 2.0 * ke / (n_dof * KB)


def minimize_energy(
    system: CGSystem,
    max_steps: int = 1000,
    step_size: float = 0.0001,
    force_cap: float = 1000.0,
    tolerance: float = 1.0,
    verbose: bool = True,
) -> float:
    """Steepest descent energy minimization with force capping.

    Uses Rust backend if available.
    """
    if _HAS_RUST:
        if verbose:
            _, energies = system.compute_forces()
            print(f"Minimizing energy (initial PE={energies['potential']:.1f} kJ/mol) [Rust]")
        t = system._get_topo_arrays()
        final_pe = _rs.minimize_energy_rs(
            system.positions,
            t["bond_i"], t["bond_j"], t["bond_r0"], t["bond_k"],
            t["angle_i"], t["angle_j"], t["angle_k"],
            t["angle_theta0"], t["angle_k_param"],
            t["dih_i"], t["dih_j"], t["dih_k"], t["dih_l"],
            t["dih_phi0"], t["dih_k_param"], t["dih_n"],
            t["pair_i"], t["pair_j"], t["pair_sigma"], t["pair_epsilon"],
            1.5, max_steps, step_size, force_cap, tolerance,
        )
        if verbose:
            print(f"  Finished {max_steps} steps: PE={final_pe:.1f} [Rust]")
        return final_pe

    if verbose:
        forces, energies = system.compute_forces()
        print(f"Minimizing energy (initial PE={energies['potential']:.1f} kJ/mol)")

    for step in range(max_steps):
        forces, energies = system.compute_forces()

        # Cap forces
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = force_norms.max()

        if max_force < tolerance:
            if verbose:
                print(f"  Converged at step {step}: "
                      f"PE={energies['potential']:.1f}, max_F={max_force:.2f}")
            return energies["potential"]

        # Cap individual forces
        too_large = force_norms > force_cap
        if too_large.any():
            scale = np.where(too_large, force_cap / force_norms, 1.0)
            forces *= scale[:, None]

        # Steepest descent step: normalize force direction, fixed step size
        for bi in range(system.n_beads):
            fn = force_norms[bi]
            if fn > 1e-12:
                system.positions[bi] += step_size * forces[bi] / fn

        if verbose and (step + 1) % 200 == 0:
            print(f"  Step {step+1}: PE={energies['potential']:.1f}, "
                  f"max_F={max_force:.1f}")

    forces, energies = system.compute_forces()
    if verbose:
        print(f"  Finished {max_steps} steps: PE={energies['potential']:.1f}")
    return energies["potential"]


def compute_native_contacts(
    atom_positions: np.ndarray,
    cutoff: float = 0.60,
) -> list[tuple[int, int]]:
    """Compute native atom-atom contacts from reference structure.

    Called once at startup. Returns list of (atom_i, atom_j) pairs
    that are within cutoff in the reference PDB structure.
    This contact list is a property of the molecule, not the mapping.
    """
    from scipy.spatial.distance import cdist
    dmat = cdist(atom_positions, atom_positions)
    contacts = []
    n = len(atom_positions)
    for i in range(n):
        for j in range(i + 1, n):
            if dmat[i, j] < cutoff:
                contacts.append((i, j))
    return contacts


def project_contacts_to_beads(
    native_contacts: list[tuple[int, int]],
    mapping: list[list[int]],
    ref_atom_positions: np.ndarray,
    atom_masses: np.ndarray,
    bond_list: list,
    base_epsilon: float = 10.0,
    max_epsilon: float = 25.0,
) -> tuple[list[tuple[int, int]], list['LJParam']]:
    """Project atom-level contacts to bead-level Go contacts.

    For each pair of beads, counts how many native atom contacts exist
    between their constituent atoms. More shared contacts = stronger
    Go attraction. Reference distance from PDB bead COM positions.

    This is O(n_contacts) — just integer lookups, microseconds.
    """
    n_atoms = len(ref_atom_positions)
    n_beads = len(mapping)

    # Build atom → bead lookup
    atom_to_bead = np.full(n_atoms, -1, dtype=int)
    for bead_idx, group in enumerate(mapping):
        for a in group:
            atom_to_bead[a] = bead_idx

    # Count contacts per bead pair
    from collections import Counter
    pair_counts = Counter()
    for a, b in native_contacts:
        bi = atom_to_bead[a]
        bj = atom_to_bead[b]
        if bi < 0 or bj < 0 or bi == bj:
            continue
        pair = (min(bi, bj), max(bi, bj))
        pair_counts[pair] += 1

    # Exclude directly bonded bead pairs
    bonded = {(min(i, j), max(i, j)) for i, j in bond_list}

    # Compute reference bead COM positions
    bead_com = np.zeros((n_beads, 3))
    for i, group in enumerate(mapping):
        grp = np.asarray(group)
        m = atom_masses[grp]
        bead_com[i] = (m[:, None] * ref_atom_positions[grp]).sum(axis=0) / m.sum()

    # Create Go contacts scaled by contact count
    max_count = max(pair_counts.values()) if pair_counts else 1
    go_pairs = []
    go_params = []
    for (bi, bj), count in pair_counts.items():
        if (bi, bj) in bonded:
            continue
        r0 = np.linalg.norm(bead_com[bi] - bead_com[bj])
        if r0 < 1e-6:
            continue
        sigma = r0 / (2.0 ** (1.0 / 6.0))
        # Scale epsilon by fraction of max contacts
        eps = base_epsilon + (max_epsilon - base_epsilon) * (count / max_count)
        go_pairs.append((bi, bj))
        go_params.append(LJParam(sigma=sigma, epsilon=eps))

    return go_pairs, go_params


def _add_structure_bias(
    positions: np.ndarray,
    bond_list: list,
    bond_params: list,
    nb_pairs: list,
    nb_params: list,
    exclude_set: set,
    mode: str = "none",
    en_cutoff: float = 0.9,
    en_k: float = 500.0,
    native_contacts: list[tuple[int, int]] | None = None,
    mapping: list[list[int]] | None = None,
    ref_atom_positions: np.ndarray | None = None,
    atom_masses: np.ndarray | None = None,
    verbose: bool = True,
) -> None:
    """Add structure bias to prevent entropy-driven unfolding.

    Modes:
    - 'elastic': Harmonic springs between nearby beads (ElNeDyn).
    - 'go': Atom-level native contacts projected to bead-level LJ.
      Consistent across remaps because atom contacts are fixed.
    - 'none': No bias.
    """
    if mode == "none":
        return

    n_added = 0

    if mode == "elastic":
        from scipy.spatial.distance import cdist
        dmat = cdist(positions, positions)
        for i in range(positions.shape[0]):
            for j in range(i + 1, positions.shape[0]):
                if (i, j) in exclude_set:
                    continue
                r0 = dmat[i, j]
                if r0 < en_cutoff:
                    bond_list.append((i, j))
                    bond_params.append(BondParam(r0=r0, k=en_k))
                    n_added += 1

    elif mode == "go":
        if native_contacts is None or mapping is None or ref_atom_positions is None:
            raise ValueError(
                "Go contacts require native_contacts, mapping, "
                "ref_atom_positions, and atom_masses"
            )
        go_pairs, go_params = project_contacts_to_beads(
            native_contacts, mapping, ref_atom_positions,
            atom_masses, bond_list,
        )
        nb_pairs.extend(go_pairs)
        nb_params.extend(go_params)
        n_added = len(go_pairs)

    if verbose and n_added > 0:
        print(f"  Structure bias ({mode}): {n_added} contacts")


def setup_cg_system(
    pdb_path: Path,
    ff_path: Path,
    n_beads: int | None = None,
    ratio: int = 4,
    temperature: float = 300.0,
    bond_scale: float = 1.0,
    angle_scale: float = 1.0,
    dihedral_scale: float = 1.0,
    structure_bias: str = "none",
    verbose: bool = True,
) -> CGSystem:
    """Set up a CG system from a PDB file and force field.

    1. Load heavy atoms
    2. K-means mapping → bead positions + masses
    3. Classify beads → look up force field parameters
    4. Initialize velocities from Maxwell-Boltzmann

    Parameters
    ----------
    pdb_path : Path
        Original PDB file.
    ff_path : Path
        Path to cg_forcefield.json.
    n_beads : int or None
        Number of CG beads. If None, uses n_atoms // ratio.
    ratio : int
        Default atom-to-bead ratio.
    temperature : float
        Initial temperature for velocity initialization (K).
    verbose : bool
        Print progress.

    Returns
    -------
    CGSystem
    """
    import warnings
    import MDAnalysis as mda
    from adaptive_cg.core.molecule import WATER_RESNAMES, ION_RESNAMES, _detect_mol_type
    from adaptive_cg.core.strategies import kmeans_mapping
    from adaptive_cg.core.extract import (
        classify_bead, detect_bonds, detect_angles, compute_bead_positions,
    )

    # Load force field
    ff = load_forcefield(ff_path)
    if verbose:
        print(f"Force field: {len(ff['bonds'])} bond types, "
              f"{len(ff['angles'])} angle types, "
              f"{len(ff['nonbonded'])} non-bonded types")

    # Load heavy atoms
    pdb_id = pdb_path.stem.upper()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(str(pdb_path))

    exclude_list = " ".join(sorted(WATER_RESNAMES | ION_RESNAMES))
    sel_string = f"not type H and not resname {exclude_list}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ag = u.select_atoms(sel_string)

    chains = sorted(set(ag.segids))
    if len(chains) > 1:
        ag = ag.select_atoms(f"segid {chains[0]}")

    atom_positions = ag.positions / 10.0  # Angstroms → nm
    atom_masses = ag.masses.copy()
    nonzero = atom_masses > 0
    if not nonzero.all():
        atom_positions = atom_positions[nonzero]
        atom_masses = atom_masses[nonzero]
        ag = ag[nonzero]

    n_atoms = len(ag)
    elements = [a.element.strip() if a.element else a.name.strip()[0] for a in ag]
    atom_names = [a.name.strip() for a in ag]
    residue_names = [a.resname.strip() for a in ag]
    mol_type = _detect_mol_type(residue_names)

    if verbose:
        print(f"Molecule: {pdb_id}, {n_atoms} heavy atoms, type={mol_type}")

    # K-means mapping
    if n_beads is None:
        n_beads = max(2, n_atoms // ratio)
    mapping = kmeans_mapping(atom_positions, atom_masses, n_beads)
    n_beads = len(mapping)  # may be fewer after merging small clusters
    if verbose:
        sizes = [len(g) for g in mapping]
        print(f"Mapping: {len(mapping)} beads, "
              f"sizes {min(sizes)}-{max(sizes)} atoms/bead")

    # Bead positions and masses
    bead_positions = compute_bead_positions(mapping, atom_positions, atom_masses)
    bead_masses = np.array([atom_masses[g].sum() for g in mapping])

    # Classify beads
    bead_keys = []
    for group in mapping:
        bc = classify_bead(group, elements, atom_names, residue_names, mol_type)
        bead_keys.append(bc.key)

    # Detect topology
    from adaptive_cg.core.extract import detect_dihedrals
    bonds = detect_bonds(mapping, n_atoms)
    angles = detect_angles(bonds)
    dihedrals = detect_dihedrals(bonds)
    if verbose:
        print(f"Topology: {len(bonds)} bonds, {len(angles)} angles, "
              f"{len(dihedrals)} dihedrals")

    # Match force field parameters
    bond_list = []
    bond_params = []
    n_bond_missing = 0
    for i, j in bonds:
        key = "--".join(sorted([bead_keys[i], bead_keys[j]]))
        if key in ff["bonds"]:
            bond_list.append((i, j))
            p = ff["bonds"][key]
            bond_params.append(BondParam(r0=p.r0, k=p.k * bond_scale))
        else:
            # Default bond: use actual distance, moderate force constant
            r0 = float(np.linalg.norm(bead_positions[i] - bead_positions[j]))
            bond_list.append((i, j))
            bond_params.append(BondParam(r0=r0, k=200.0 * bond_scale))
            n_bond_missing += 1

    angle_list_out = []
    angle_params = []
    n_angle_missing = 0
    for i, j, k in angles:
        key = "--".join([bead_keys[i], bead_keys[j], bead_keys[k]])
        key_rev = "--".join([bead_keys[k], bead_keys[j], bead_keys[i]])
        p = ff["angles"].get(key) or ff["angles"].get(key_rev)
        if p:
            angle_list_out.append((i, j, k))
            angle_params.append(AngleParam(theta0=p.theta0, k=p.k * angle_scale))
        else:
            # Default angle: use actual angle, moderate force constant
            v1 = bead_positions[i] - bead_positions[j]
            v2 = bead_positions[k] - bead_positions[j]
            cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
            theta0 = float(np.arccos(np.clip(cos_t, -1, 1)))
            angle_list_out.append((i, j, k))
            angle_params.append(AngleParam(theta0=theta0, k=20.0 * angle_scale))
            n_angle_missing += 1

    # Match dihedrals
    dihedral_list = []
    dihedral_params = []
    n_dih_missing = 0
    for i, j, k, l in dihedrals:
        key = "--".join([bead_keys[i], bead_keys[j], bead_keys[k], bead_keys[l]])
        key_rev = "--".join([bead_keys[l], bead_keys[k], bead_keys[j], bead_keys[i]])
        p = ff["dihedrals"].get(key) or ff["dihedrals"].get(key_rev)
        if p:
            dihedral_list.append((i, j, k, l))
            dihedral_params.append(DihedralParam(phi0=p.phi0, k=p.k * dihedral_scale, n=p.n))
        else:
            n_dih_missing += 1

    # Non-bonded pairs: exclude 1-2, 1-3, and 1-4 neighbors.
    from collections import defaultdict
    neighbors = defaultdict(set)
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)

    exclude_set = set()
    for i in range(n_beads):
        # 1-2: direct bond partners
        for j in neighbors[i]:
            exclude_set.add((min(i, j), max(i, j)))
            # 1-3: partners of partners
            for k in neighbors[j]:
                if k != i:
                    exclude_set.add((min(i, k), max(i, k)))
                    # 1-4: one more hop
                    for l in neighbors[k]:
                        if l != i and l != j:
                            exclude_set.add((min(i, l), max(i, l)))

    # Compute per-bead effective radius from number of atoms.
    # Assume each heavy atom occupies ~0.15 nm radius; bead radius
    # scales as n_atoms^(1/3) for a roughly spherical cluster.
    bead_radii = np.array([
        0.15 * len(g) ** (1.0 / 3.0) for g in mapping
    ])

    nb_pairs = []
    nb_params_list = []
    n_nb_missing = 0
    default_epsilon = 2.0  # kJ/mol — weak attraction, gentle repulsion
    for i in range(n_beads):
        for j in range(i + 1, n_beads):
            if (i, j) in exclude_set:
                continue
            # Derive sigma from bead radii (Lorentz combining rule)
            sigma_ij = bead_radii[i] + bead_radii[j]
            # Use force field epsilon if available, otherwise default
            key = "--".join(sorted([bead_keys[i], bead_keys[j]]))
            if key in ff["nonbonded"]:
                eps = ff["nonbonded"][key].epsilon
            else:
                eps = default_epsilon
                n_nb_missing += 1
            nb_pairs.append((i, j))
            nb_params_list.append(LJParam(sigma=sigma_ij, epsilon=eps))

    if verbose:
        print(f"Parameters matched: {len(bond_list)} bonds, "
              f"{len(angle_list_out)} angles, {len(dihedral_list)} dihedrals, "
              f"{len(nb_pairs)} non-bonded pairs")
        scales = []
        if bond_scale != 1.0: scales.append(f"bond={bond_scale}")
        if angle_scale != 1.0: scales.append(f"angle={angle_scale}")
        if dihedral_scale != 1.0: scales.append(f"dihedral={dihedral_scale}")
        if scales:
            print(f"  Scaling: {', '.join(scales)}")
        if n_bond_missing or n_angle_missing or n_dih_missing or n_nb_missing:
            print(f"  Missing: {n_bond_missing} bonds, "
                  f"{n_angle_missing} angles, {n_dih_missing} dihedrals, "
                  f"{n_nb_missing} non-bonded")

    # Structure bias (elastic network or Go contacts)
    native_contacts = None
    if structure_bias == "go":
        native_contacts = compute_native_contacts(atom_positions)
        if verbose:
            print(f"  Native atom contacts: {len(native_contacts)}")
    _add_structure_bias(
        bead_positions, bond_list, bond_params,
        nb_pairs, nb_params_list, exclude_set,
        mode=structure_bias, verbose=verbose,
        native_contacts=native_contacts,
        mapping=mapping,
        ref_atom_positions=atom_positions,
        atom_masses=atom_masses,
    )

    # Initialize velocities (Maxwell-Boltzmann)
    velocities = np.zeros((n_beads, 3))
    for i in range(n_beads):
        sigma_v = np.sqrt(KB * temperature / bead_masses[i])
        velocities[i] = np.random.normal(0.0, sigma_v, size=3)

    # Remove center-of-mass velocity
    total_mass = bead_masses.sum()
    com_vel = (bead_masses[:, None] * velocities).sum(axis=0) / total_mass
    velocities -= com_vel

    if verbose:
        ke = 0.5 * np.sum(bead_masses[:, None] * velocities ** 2)
        n_dof = max(3 * n_beads - 3, 1)
        init_temp = 2.0 * ke / (n_dof * KB)
        print(f"Initial temperature: {init_temp:.1f} K")

    return CGSystem(
        n_beads=n_beads,
        positions=bead_positions,
        velocities=velocities,
        masses=bead_masses,
        bead_class_keys=bead_keys,
        bond_list=bond_list,
        bond_params=bond_params,
        angle_list=angle_list_out,
        angle_params=angle_params,
        dihedral_list=dihedral_list,
        dihedral_params=dihedral_params,
        nb_pairs=nb_pairs,
        nb_params=nb_params_list,
    )


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------

def velocity_verlet_step(
    system: CGSystem,
    forces: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One velocity Verlet integration step.

    Updates positions and velocities in-place.
    Returns new forces after position update.

    Parameters
    ----------
    system : CGSystem
    forces : np.ndarray, shape (n_beads, 3)
        Current forces on each bead (kJ/mol/nm).
    dt : float
        Timestep in ps.

    Returns
    -------
    new_forces : np.ndarray
    """
    m = system.masses[:, None]  # (n, 1) for broadcasting
    acc = forces / m

    # Half-step velocity
    system.velocities += 0.5 * acc * dt

    # Full-step position
    system.positions += system.velocities * dt

    # New forces at updated positions
    new_forces, _ = system.compute_forces()
    new_acc = new_forces / m

    # Complete velocity step
    system.velocities += 0.5 * new_acc * dt

    return new_forces


def langevin_step(
    system: CGSystem,
    forces: np.ndarray,
    dt: float,
    temperature: float,
    friction: float = 1.0,
) -> np.ndarray:
    """One Langevin dynamics step (BAOAB splitting).

    Uses Rust backend if available (100x+ faster).
    """
    if _HAS_RUST:
        t = system._get_topo_arrays()
        new_f, _, _, _, _, _ = _rs.langevin_step_rs(
            system.positions, system.velocities, system.masses,
            forces, dt, temperature, friction,
            t["bond_i"], t["bond_j"], t["bond_r0"], t["bond_k"],
            t["angle_i"], t["angle_j"], t["angle_k"],
            t["angle_theta0"], t["angle_k_param"],
            t["dih_i"], t["dih_j"], t["dih_k"], t["dih_l"],
            t["dih_phi0"], t["dih_k_param"], t["dih_n"],
            t["pair_i"], t["pair_j"], t["pair_sigma"], t["pair_epsilon"],
            1.5,
        )
        return np.asarray(new_f)

    m = system.masses[:, None]
    half_dt = 0.5 * dt

    system.velocities += half_dt * forces / m
    system.positions += half_dt * system.velocities

    c1 = np.exp(-friction * dt)
    c2 = np.sqrt((1.0 - c1 * c1) * KB * temperature)
    for i in range(system.n_beads):
        sigma = c2 / np.sqrt(system.masses[i])
        system.velocities[i] = (
            c1 * system.velocities[i]
            + sigma * np.random.normal(size=3)
        )

    system.positions += half_dt * system.velocities
    new_forces, _ = system.compute_forces()
    system.velocities += half_dt * new_forces / m

    return new_forces


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

@dataclass
class SimulationLog:
    """Recorded quantities from a CG simulation."""
    steps: list[int] = field(default_factory=list)
    time_ps: list[float] = field(default_factory=list)
    temperature: list[float] = field(default_factory=list)
    kinetic_energy: list[float] = field(default_factory=list)
    potential_energy: list[float] = field(default_factory=list)
    total_energy: list[float] = field(default_factory=list)
    bond_energy: list[float] = field(default_factory=list)
    angle_energy: list[float] = field(default_factory=list)
    nonbonded_energy: list[float] = field(default_factory=list)

    def save(self, path: Path):
        """Save log as CSV."""
        import csv
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "time_ps", "temperature", "kinetic_energy",
                "potential_energy", "total_energy",
                "bond_energy", "angle_energy", "nonbonded_energy",
            ])
            for i in range(len(self.steps)):
                writer.writerow([
                    self.steps[i], f"{self.time_ps[i]:.4f}",
                    f"{self.temperature[i]:.2f}",
                    f"{self.kinetic_energy[i]:.4f}",
                    f"{self.potential_energy[i]:.4f}",
                    f"{self.total_energy[i]:.4f}",
                    f"{self.bond_energy[i]:.4f}",
                    f"{self.angle_energy[i]:.4f}",
                    f"{self.nonbonded_energy[i]:.4f}",
                ])


def run_cg_simulation(
    system: CGSystem,
    n_steps: int = 10000,
    dt: float = 0.01,
    temperature: float = 300.0,
    friction: float = 1.0,
    thermostat: str = "langevin",
    log_interval: int = 100,
    save_interval: int = 1000,
    trajectory_path: Path | None = None,
    verbose: bool = True,
) -> SimulationLog:
    """Run a CG molecular dynamics simulation.

    Parameters
    ----------
    system : CGSystem
        The CG system to simulate.
    n_steps : int
        Number of integration steps.
    dt : float
        Timestep in ps.
    temperature : float
        Target temperature in Kelvin.
    friction : float
        Langevin friction coefficient (1/ps).
    thermostat : str
        "langevin" or "nve" (no thermostat).
    log_interval : int
        Log energies every N steps.
    save_interval : int
        Save positions to trajectory every N steps.
    trajectory_path : Path or None
        If set, save trajectory frames as numpy .npy file.
    verbose : bool
        Print progress.

    Returns
    -------
    SimulationLog
    """
    log = SimulationLog()
    frames = []

    # Energy minimization before MD
    minimize_energy(system, max_steps=5000, step_size=0.001, verbose=verbose)

    # Re-initialize velocities after minimization (positions changed)
    for i in range(system.n_beads):
        sigma_v = np.sqrt(KB * temperature / system.masses[i])
        system.velocities[i] = np.random.normal(0.0, sigma_v, size=3)
    total_mass = system.masses.sum()
    com_vel = (system.masses[:, None] * system.velocities).sum(axis=0) / total_mass
    system.velocities -= com_vel

    # Initial forces
    forces, energies = system.compute_forces()

    if verbose:
        total_time = n_steps * dt
        print(f"Running CG MD: {n_steps} steps, dt={dt} ps, "
              f"total={total_time:.2f} ps")
        print(f"Thermostat: {thermostat}, T={temperature} K")

    for step in range(n_steps):
        # Integrate
        if thermostat == "langevin":
            forces = langevin_step(system, forces, dt, temperature, friction)
        else:
            forces = velocity_verlet_step(system, forces, dt)

        # Log
        if (step + 1) % log_interval == 0:
            _, energies = system.compute_forces()
            ke = system.kinetic_energy()
            pe = energies["potential"]
            temp = system.temperature()

            log.steps.append(step + 1)
            log.time_ps.append((step + 1) * dt)
            log.temperature.append(temp)
            log.kinetic_energy.append(ke)
            log.potential_energy.append(pe)
            log.total_energy.append(ke + pe)
            log.bond_energy.append(energies["bond"])
            log.angle_energy.append(energies["angle"])
            log.nonbonded_energy.append(energies["nonbonded"])

            if verbose and (step + 1) % (log_interval * 10) == 0:
                print(f"  Step {step+1}/{n_steps}: T={temp:.1f} K, "
                      f"PE={pe:.1f}, KE={ke:.1f}, "
                      f"E_total={ke+pe:.1f} kJ/mol")

        # Save frame
        if save_interval and (step + 1) % save_interval == 0:
            frames.append(system.positions.copy())

    # Save trajectory
    if trajectory_path and frames:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(trajectory_path, np.stack(frames))
        if verbose:
            print(f"Saved {len(frames)} frames to {trajectory_path}")

    if verbose:
        print("Done.")

    return log
