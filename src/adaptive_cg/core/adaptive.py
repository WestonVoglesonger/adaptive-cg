"""Adaptive resolution controller for CG molecular dynamics.

Monitors per-region activity during simulation and dynamically
adjusts bead resolution — more beads where the molecule is active,
fewer where it's static.

Strategies:
1. Fixed (baseline): no remapping, use run_cg_simulation directly
2. Periodic: check activity every N steps, remap if distribution shifted
3. Continuous (AdResS-style): future work

The adaptive controller is the core contribution of this project.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from adaptive_cg.core.engine import (
    CGSystem, KB, LJParam, SimulationLog,
    load_forcefield, minimize_energy, langevin_step, velocity_verlet_step,
)


# ---------------------------------------------------------------------------
# Activity monitoring
# ---------------------------------------------------------------------------

@dataclass
class ActivityMonitor:
    """Tracks per-bead position fluctuations over a sliding window.

    RMSF (root-mean-square fluctuation) measures how much each bead
    deviates from its mean position. High RMSF = active, low = static.
    """
    window_size: int = 50
    _positions: deque = field(default_factory=deque)

    def record(self, positions: np.ndarray):
        """Record a snapshot of bead positions."""
        self._positions.append(positions.copy())
        if len(self._positions) > self.window_size:
            self._positions.popleft()

    @property
    def n_samples(self) -> int:
        return len(self._positions)

    def rmsf(self) -> np.ndarray:
        """Per-bead RMSF in nm. Shape (n_beads,)."""
        if len(self._positions) < 2:
            n = self._positions[0].shape[0] if self._positions else 0
            return np.zeros(n)
        stacked = np.stack(list(self._positions))  # (frames, beads, 3)
        mean_pos = stacked.mean(axis=0)
        msf = ((stacked - mean_pos[None]) ** 2).sum(axis=2).mean(axis=0)
        return np.sqrt(msf)

    def reset(self):
        self._positions.clear()


# ---------------------------------------------------------------------------
# Region utilities
# ---------------------------------------------------------------------------

def partition_atoms(n_atoms: int, n_regions: int) -> list[tuple[int, int]]:
    """Divide atoms into N sequential regions along the chain.

    Returns list of (start_atom, end_atom) index ranges.
    Sequential regions correspond to different structural domains
    in proteins (N-terminal, core, C-terminal, etc.).
    """
    per = n_atoms // n_regions
    regions = []
    for i in range(n_regions):
        start = i * per
        end = (i + 1) * per if i < n_regions - 1 else n_atoms
        regions.append((start, end))
    return regions


def assign_beads_to_regions(
    mapping: list[list[int]],
    regions: list[tuple[int, int]],
) -> np.ndarray:
    """Assign each bead to its majority region.

    A bead belongs to the region containing most of its atoms.
    After per-region remapping, beads are always fully within one region.

    Returns array of region indices, shape (n_beads,).
    """
    n_beads = len(mapping)
    assignments = np.zeros(n_beads, dtype=int)
    for bead_idx, group in enumerate(mapping):
        best_r, best_count = 0, 0
        for r_idx, (start, end) in enumerate(regions):
            count = sum(1 for a in group if start <= a < end)
            if count > best_count:
                best_count = count
                best_r = r_idx
        assignments[bead_idx] = best_r
    return assignments


def compute_region_activity(
    rmsf: np.ndarray,
    bead_regions: np.ndarray,
    n_regions: int,
) -> np.ndarray:
    """Per-region activity = mean RMSF of beads in that region."""
    scores = np.zeros(n_regions)
    for r in range(n_regions):
        mask = bead_regions == r
        if mask.any():
            scores[r] = rmsf[mask].mean()
    return scores


# ---------------------------------------------------------------------------
# Bead allocation
# ---------------------------------------------------------------------------

def allocate_beads(
    activity: np.ndarray,
    region_sizes: np.ndarray,
    total_beads: int,
    min_per_region: int = 2,
    activity_weight: float = 0.5,
    current_alloc: np.ndarray | None = None,
    max_change: int = 5,
) -> np.ndarray:
    """Allocate beads to regions based on activity and size.

    allocation ~ (1-w) * region_size + w * activity

    w=0: beads proportional to atom count (uniform density).
    w=1: purely activity-driven allocation.

    If current_alloc is provided, caps per-region change to max_change
    to avoid drastic remaps that cause temperature spikes.
    """
    n = len(activity)
    size_norm = region_sizes / region_sizes.sum()
    act_sum = activity.sum()
    act_norm = activity / act_sum if act_sum > 0 else np.ones(n) / n

    weights = (1 - activity_weight) * size_norm + activity_weight * act_norm

    raw = weights / weights.sum() * total_beads
    alloc = np.maximum(np.round(raw).astype(int), min_per_region)

    # Cap per-region change to avoid drastic remaps
    if current_alloc is not None:
        alloc = np.clip(
            alloc,
            np.maximum(current_alloc - max_change, min_per_region),
            current_alloc + max_change,
        )

    # Fix total to exactly match, respecting caps
    diff = total_beads - alloc.sum()
    if diff != 0 and current_alloc is not None:
        upper = current_alloc + max_change
        lower = np.maximum(current_alloc - max_change, min_per_region)
        order = np.argsort(-activity) if diff > 0 else np.argsort(activity)
        passes = 0
        initial_diff = abs(diff)
        while diff != 0 and passes < n * initial_diff + n:
            idx = order[passes % n]
            if diff > 0 and alloc[idx] < upper[idx]:
                alloc[idx] += 1
                diff -= 1
            elif diff < 0 and alloc[idx] > lower[idx]:
                alloc[idx] -= 1
                diff += 1
            passes += 1
    elif diff != 0:
        order = np.argsort(-activity) if diff > 0 else np.argsort(activity)
        for i in range(abs(diff)):
            idx = order[i % n]
            if diff > 0:
                alloc[idx] += 1
            elif alloc[idx] > min_per_region:
                alloc[idx] -= 1

    return alloc


# ---------------------------------------------------------------------------
# Atom position estimation
# ---------------------------------------------------------------------------

def estimate_atom_positions(
    mapping: list[list[int]],
    bead_positions: np.ndarray,
    original_atom_positions: np.ndarray,
    atom_masses: np.ndarray,
) -> np.ndarray:
    """Estimate current atom positions from CG bead positions.

    Each atom keeps its offset from the original bead COM,
    but the COM is shifted to the current simulated bead position.
    Preserves intra-bead structure while reflecting dynamics.
    """
    estimated = original_atom_positions.copy()
    for bead_idx, group in enumerate(mapping):
        grp = np.asarray(group)
        m = atom_masses[grp]
        orig_com = (m[:, None] * original_atom_positions[grp]).sum(axis=0) / m.sum()
        disp = bead_positions[bead_idx] - orig_com
        for a in group:
            estimated[a] = original_atom_positions[a] + disp
    return estimated


# ---------------------------------------------------------------------------
# System remapping
# ---------------------------------------------------------------------------

def remap_system(
    atom_positions: np.ndarray,
    atom_masses: np.ndarray,
    elements: list[str],
    atom_names: list[str],
    residue_names: list[str],
    mol_type: str,
    regions: list[tuple[int, int]],
    allocation: np.ndarray,
    ff: dict,
    temperature: float = 300.0,
    dihedral_scale: float = 1.0,
    structure_bias: str = "none",
) -> tuple[CGSystem, list[list[int]]]:
    """Create a new CGSystem with per-region bead allocation.

    Runs k-means independently within each region, then combines
    into a global mapping with full topology and force field matching.

    Returns (system, mapping).
    """
    from adaptive_cg.core.strategies import kmeans_mapping
    from adaptive_cg.core.extract import (
        classify_bead, detect_bonds, detect_angles, detect_dihedrals,
        compute_bead_positions,
    )

    n_atoms = atom_positions.shape[0]

    # Per-region k-means
    global_mapping = []
    for r_idx, (start, end) in enumerate(regions):
        n_beads_region = int(allocation[r_idx])
        n_region = end - start

        if n_beads_region >= n_region:
            # More beads than atoms — one atom per bead
            for a in range(start, end):
                global_mapping.append([a])
        else:
            region_pos = atom_positions[start:end]
            region_masses = atom_masses[start:end]
            local_mapping = kmeans_mapping(region_pos, region_masses, n_beads_region)
            # Translate local indices to global
            for group in local_mapping:
                global_mapping.append([start + i for i in group])

    n_beads = len(global_mapping)

    # Bead positions (COM) and masses
    bead_positions = compute_bead_positions(global_mapping, atom_positions, atom_masses)
    bead_masses = np.array([atom_masses[g].sum() for g in global_mapping])

    # Classify beads
    bead_keys = []
    for group in global_mapping:
        bc = classify_bead(group, elements, atom_names, residue_names, mol_type)
        bead_keys.append(bc.key)

    # Topology
    bonds = detect_bonds(global_mapping, n_atoms)
    angles = detect_angles(bonds)
    dihedrals = detect_dihedrals(bonds)

    # Match force field — bonds
    bond_list, bond_params = [], []
    for i, j in bonds:
        key = "--".join(sorted([bead_keys[i], bead_keys[j]]))
        if key in ff["bonds"]:
            bond_list.append((i, j))
            bond_params.append(ff["bonds"][key])

    # Match force field — angles
    angle_list, angle_params = [], []
    for i, j, k in angles:
        key = "--".join([bead_keys[i], bead_keys[j], bead_keys[k]])
        key_rev = "--".join([bead_keys[k], bead_keys[j], bead_keys[i]])
        if key in ff["angles"]:
            angle_list.append((i, j, k))
            angle_params.append(ff["angles"][key])
        elif key_rev in ff["angles"]:
            angle_list.append((i, j, k))
            angle_params.append(ff["angles"][key_rev])

    # Match force field — dihedrals
    from adaptive_cg.core.engine import DihedralParam
    dihedral_list, dihedral_params = [], []
    for i, j, k, l in dihedrals:
        key = "--".join([bead_keys[i], bead_keys[j], bead_keys[k], bead_keys[l]])
        key_rev = "--".join([bead_keys[l], bead_keys[k], bead_keys[j], bead_keys[i]])
        p = ff.get("dihedrals", {}).get(key) or ff.get("dihedrals", {}).get(key_rev)
        if p:
            dihedral_list.append((i, j, k, l))
            dihedral_params.append(DihedralParam(phi0=p.phi0, k=p.k * dihedral_scale, n=p.n))

    # Non-bonded exclusions: 1-2, 1-3, 1-4
    neighbors = defaultdict(set)
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)

    exclude_set = set()
    for i in range(n_beads):
        for j in neighbors[i]:
            exclude_set.add((min(i, j), max(i, j)))
            for k in neighbors[j]:
                if k != i:
                    exclude_set.add((min(i, k), max(i, k)))
                    for nb in neighbors[k]:
                        if nb != i and nb != j:
                            exclude_set.add((min(i, nb), max(i, nb)))

    # Non-bonded pairs with size-based sigma
    bead_radii = np.array([0.15 * len(g) ** (1.0 / 3.0) for g in global_mapping])
    nb_pairs, nb_params = [], []
    default_eps = 2.0
    for i in range(n_beads):
        for j in range(i + 1, n_beads):
            if (i, j) in exclude_set:
                continue
            sigma_ij = bead_radii[i] + bead_radii[j]
            key = "--".join(sorted([bead_keys[i], bead_keys[j]]))
            eps = ff["nonbonded"][key].epsilon if key in ff["nonbonded"] else default_eps
            nb_pairs.append((i, j))
            nb_params.append(LJParam(sigma=sigma_ij, epsilon=eps))

    # Structure bias (Go contacts or elastic network)
    from adaptive_cg.core.engine import _add_structure_bias, BondParam
    _add_structure_bias(
        bead_positions, bond_list, bond_params,
        nb_pairs, nb_params, exclude_set,
        mode=structure_bias,
    )

    # Initialize velocities (Maxwell-Boltzmann)
    velocities = np.zeros((n_beads, 3))
    for i in range(n_beads):
        sigma_v = np.sqrt(KB * temperature / bead_masses[i])
        velocities[i] = np.random.normal(0.0, sigma_v, size=3)
    com_vel = (bead_masses[:, None] * velocities).sum(axis=0) / bead_masses.sum()
    velocities -= com_vel

    system = CGSystem(
        n_beads=n_beads,
        positions=bead_positions,
        velocities=velocities,
        masses=bead_masses,
        bead_class_keys=bead_keys,
        bond_list=bond_list,
        bond_params=bond_params,
        angle_list=angle_list,
        angle_params=angle_params,
        dihedral_list=dihedral_list,
        dihedral_params=dihedral_params,
        nb_pairs=nb_pairs,
        nb_params=nb_params,
    )

    return system, global_mapping


# ---------------------------------------------------------------------------
# Adaptive simulation log
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveLog(SimulationLog):
    """Extended log with remapping events."""
    remap_steps: list[int] = field(default_factory=list)
    remap_old_beads: list[int] = field(default_factory=list)
    remap_new_beads: list[int] = field(default_factory=list)
    region_activity_history: list[list[float]] = field(default_factory=list)

    def save_adaptive(self, path: Path):
        """Save adaptive-specific data as JSON."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "remap_steps": self.remap_steps,
            "remap_old_beads": self.remap_old_beads,
            "remap_new_beads": self.remap_new_beads,
            "region_activity_history": self.region_activity_history,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Adaptive simulation runner
# ---------------------------------------------------------------------------

def should_remap(
    current: np.ndarray,
    proposed: np.ndarray,
    threshold: int = 3,
) -> bool:
    """True if any region's allocation differs by >= threshold beads."""
    return int(np.abs(current - proposed).max()) >= threshold


def run_adaptive_simulation(
    pdb_path: Path,
    ff_path: Path,
    n_beads: int | None = None,
    ratio: int = 4,
    n_steps: int = 100000,
    dt: float = 0.01,
    temperature: float = 300.0,
    friction: float = 1.0,
    thermostat: str = "langevin",
    n_regions: int = 5,
    monitor_interval: int = 100,
    remap_check_interval: int = 5000,
    remap_threshold: int = 3,
    activity_weight: float = 0.5,
    monitor_window: int = 50,
    dihedral_scale: float = 1.0,
    structure_bias: str = "none",
    log_interval: int = 100,
    save_interval: int = 1000,
    trajectory_path: Path | None = None,
    verbose: bool = True,
) -> AdaptiveLog:
    """Run CG MD with periodic adaptive remapping.

    Every remap_check_interval steps:
    1. Compute per-region RMSF from sliding window
    2. Propose new bead allocation (activity-weighted)
    3. If allocation shifted enough, remap the system
    4. Minimize + re-thermalize at new resolution
    5. Continue simulation

    Total bead count stays fixed; activity redistributes beads across regions.

    Parameters
    ----------
    pdb_path : Path
        Original PDB file.
    ff_path : Path
        CG force field JSON.
    n_beads : int or None
        Total number of CG beads (fixed throughout).
    ratio : int
        Default atom-to-bead ratio if n_beads not set.
    n_steps : int
        Total integration steps.
    dt : float
        Timestep in ps.
    temperature : float
        Target temperature in K.
    friction : float
        Langevin friction coefficient (1/ps).
    thermostat : str
        "langevin" or "nve".
    n_regions : int
        Number of sequential chain regions to monitor.
    monitor_interval : int
        Steps between activity recordings.
    remap_check_interval : int
        Steps between remap decisions.
    remap_threshold : int
        Min bead difference in any region to trigger remap.
    activity_weight : float
        Weight of activity vs size in allocation (0-1).
    monitor_window : int
        Number of snapshots in RMSF sliding window.
    log_interval : int
        Steps between energy logging.
    save_interval : int
        Steps between trajectory frame saves.
    trajectory_path : Path or None
        Where to save trajectory (.npz, variable bead count).
    verbose : bool
        Print progress.

    Returns
    -------
    AdaptiveLog
    """
    import warnings
    import MDAnalysis as mda
    from adaptive_cg.core.molecule import WATER_RESNAMES, ION_RESNAMES, _detect_mol_type

    # --- Load force field ---
    ff = load_forcefield(ff_path)

    # --- Load atom-level data (kept for remapping) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(str(pdb_path))
    exclude_list = " ".join(sorted(WATER_RESNAMES | ION_RESNAMES))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ag = u.select_atoms(f"not type H and not resname {exclude_list}")
    chains = sorted(set(ag.segids))
    if len(chains) > 1:
        ag = ag.select_atoms(f"segid {chains[0]}")

    original_atoms = ag.positions / 10.0  # Angstroms -> nm
    atom_masses = ag.masses.copy()
    nonzero = atom_masses > 0
    if not nonzero.all():
        original_atoms = original_atoms[nonzero]
        atom_masses = atom_masses[nonzero]
        ag = ag[nonzero]

    n_atoms = len(ag)
    elements = [a.element.strip() if a.element else a.name.strip()[0] for a in ag]
    atom_names = [a.name.strip() for a in ag]
    residue_names = [a.resname.strip() for a in ag]
    mol_type = _detect_mol_type(residue_names)

    if n_beads is None:
        n_beads = max(2, n_atoms // ratio)

    # --- Define regions ---
    regions = partition_atoms(n_atoms, n_regions)
    region_sizes = np.array([end - start for start, end in regions])

    if verbose:
        print(f"=== Adaptive CG Simulation ===")
        print(f"Molecule: {pdb_path.stem.upper()}, {n_atoms} atoms, "
              f"{n_beads} beads, {n_regions} regions")
        print(f"Region sizes: {', '.join(str(s) for s in region_sizes)} atoms")
        print(f"Remap check every {remap_check_interval} steps, "
              f"threshold={remap_threshold}, activity_weight={activity_weight}")

    # --- Initial allocation (uniform by size) ---
    current_alloc = allocate_beads(
        np.ones(n_regions), region_sizes, n_beads, activity_weight=0.0,
    )

    # --- Build initial system ---
    system, mapping = remap_system(
        original_atoms, atom_masses, elements, atom_names, residue_names,
        mol_type, regions, current_alloc, ff, temperature, dihedral_scale, structure_bias,
    )

    if verbose:
        print(f"Initial allocation: {current_alloc} "
              f"(sum={current_alloc.sum()}, actual={system.n_beads})")
        print(f"Topology: {len(system.bond_list)} bonds, "
              f"{len(system.angle_list)} angles, "
              f"{len(system.nb_pairs)} nb pairs")

    # --- Minimize ---
    minimize_energy(system, max_steps=5000, step_size=0.001, verbose=verbose)

    # Re-init velocities after minimization
    for i in range(system.n_beads):
        sigma_v = np.sqrt(KB * temperature / system.masses[i])
        system.velocities[i] = np.random.normal(0.0, sigma_v, size=3)
    com_vel = (
        (system.masses[:, None] * system.velocities).sum(axis=0)
        / system.masses.sum()
    )
    system.velocities -= com_vel

    # --- Initialize tracking ---
    monitor = ActivityMonitor(window_size=monitor_window)
    log = AdaptiveLog()
    frames = []

    forces, _ = system.compute_forces()

    if verbose:
        total_time = n_steps * dt
        print(f"\nRunning: {n_steps} steps, dt={dt} ps "
              f"({total_time:.1f} ps total)")

    # --- Main loop ---
    for step in range(n_steps):
        # Integrate
        if thermostat == "langevin":
            forces = langevin_step(system, forces, dt, temperature, friction)
        else:
            forces = velocity_verlet_step(system, forces, dt)

        # Record for activity monitoring
        if (step + 1) % monitor_interval == 0:
            monitor.record(system.positions)

        # --- Remap check ---
        if ((step + 1) % remap_check_interval == 0
                and monitor.n_samples >= 10):
            rmsf = monitor.rmsf()
            bead_regions = assign_beads_to_regions(mapping, regions)
            reg_act = compute_region_activity(rmsf, bead_regions, n_regions)

            proposed = allocate_beads(
                reg_act, region_sizes, n_beads,
                activity_weight=activity_weight,
                current_alloc=current_alloc,
                max_change=5,
            )

            if verbose:
                print(f"\n  Step {step+1}: activity check")
                for r in range(n_regions):
                    s, e = regions[r]
                    print(f"    R{r} (atoms {s}-{e}): "
                          f"RMSF={reg_act[r]:.4f} nm  "
                          f"cur={current_alloc[r]}  prop={proposed[r]}")

            log.region_activity_history.append(reg_act.tolist())

            if should_remap(current_alloc, proposed, remap_threshold):
                old_n = system.n_beads

                # Estimate current atom positions from CG state
                est_atoms = estimate_atom_positions(
                    mapping, system.positions, original_atoms, atom_masses,
                )

                # Build new system with updated allocation
                system, mapping = remap_system(
                    est_atoms, atom_masses, elements, atom_names,
                    residue_names, mol_type, regions, proposed,
                    ff, temperature, dihedral_scale, structure_bias,
                )
                current_alloc = proposed.copy()

                # Minimize + re-thermalize
                minimize_energy(
                    system, max_steps=5000, step_size=0.001, verbose=False,
                )
                for i in range(system.n_beads):
                    sigma_v = np.sqrt(KB * temperature / system.masses[i])
                    system.velocities[i] = np.random.normal(
                        0.0, sigma_v, size=3,
                    )
                com_vel = (
                    (system.masses[:, None] * system.velocities).sum(axis=0)
                    / system.masses.sum()
                )
                system.velocities -= com_vel

                forces, _ = system.compute_forces()
                monitor.reset()

                log.remap_steps.append(step + 1)
                log.remap_old_beads.append(old_n)
                log.remap_new_beads.append(system.n_beads)

                if verbose:
                    print(f"    >>> REMAPPED: {old_n} -> {system.n_beads} beads")
                    print(f"    New allocation: {current_alloc}")
            else:
                if verbose:
                    max_d = int(np.abs(current_alloc - proposed).max())
                    print(f"    No remap needed (max diff={max_d})")

        # --- Log energies ---
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
                      f"PE={pe:.1f}, beads={system.n_beads}")

        # --- Save frame ---
        if save_interval and (step + 1) % save_interval == 0:
            frames.append(system.positions.copy())

    # --- Save trajectory (npz for variable bead counts) ---
    if trajectory_path and frames:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(trajectory_path, *frames)
        if verbose:
            print(f"\nSaved {len(frames)} frames to {trajectory_path}")

    if verbose:
        print(f"\nDone. Remapped {len(log.remap_steps)} times.")
        if log.remap_steps:
            for i, s in enumerate(log.remap_steps):
                print(f"  Step {s}: {log.remap_old_beads[i]} -> "
                      f"{log.remap_new_beads[i]} beads")

    return log
