"""All-atom MD simulation via OpenMM.

Runs short AA trajectories to generate reference data for CG force field
parameterization. Outputs trajectory (DCD) and optionally per-frame forces
(numpy .npy) for Boltzmann inversion and force matching.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    raise ImportError(
        "OpenMM is required for AA simulation. Install with:\n"
        "  pip install openmm\n"
        "or:\n"
        "  conda install -c conda-forge openmm"
    )


@dataclass
class SimulationConfig:
    """Configuration for an AA MD simulation."""

    forcefield: str = "amber14-all.xml"
    water_model: str = "amber14/tip3pfb.xml"
    padding: float = 1.0  # nm of water around solute
    ionic_strength: float = 0.15  # mol/L NaCl
    temperature: float = 300.0  # Kelvin
    pressure: float = 1.0  # atm
    timestep: float = 0.002  # ps (2 fs)
    friction: float = 1.0  # 1/ps for Langevin integrator
    nonbonded_cutoff: float = 1.0  # nm
    constraint_type: str = "HBonds"  # constrain bonds with hydrogen
    minimize_tolerance: float = 10.0  # kJ/mol/nm
    minimize_max_iter: int = 0  # 0 = until converged
    equilibration_steps: int = 25000  # 50 ps at 2 fs timestep
    production_steps: int = 500000  # 1 ns at 2 fs timestep
    save_interval: int = 1000  # save every 2 ps
    log_interval: int = 5000  # log every 10 ps


class ForceReporter:
    """Custom reporter that saves forces at each reporting interval."""

    def __init__(self, output_path: Path, report_interval: int):
        self._path = output_path
        self._interval = report_interval
        self._forces: list[np.ndarray] = []

    def describeNextReport(self, simulation):
        steps_left = self._interval - simulation.currentStep % self._interval
        return (steps_left, False, False, True, False)

    def report(self, simulation, state):
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
        self._forces.append(forces.copy())

    def save(self):
        if self._forces:
            np.save(self._path, np.stack(self._forces))


def run_aa_simulation(
    pdb_path: Path,
    output_dir: Path,
    config: SimulationConfig | None = None,
    save_forces: bool = True,
    verbose: bool = True,
) -> dict:
    """Run an all-atom MD simulation and save trajectory.

    Parameters
    ----------
    pdb_path : Path
        Path to the input PDB file.
    output_dir : Path
        Directory to write trajectory and force files.
    config : SimulationConfig or None
        Simulation parameters. Uses defaults if None.
    save_forces : bool
        Whether to save per-frame forces (for force matching).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict with keys: trajectory_path, forces_path, topology_path,
                    n_frames, n_atoms, elapsed_time_ns
    """
    if config is None:
        config = SimulationConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdb_path.stem

    if verbose:
        print(f"Loading PDB: {pdb_path}")

    # --- Load PDB ---
    pdb = app.PDBFile(str(pdb_path))

    # --- Force field ---
    if verbose:
        print(f"Setting up force field: {config.forcefield}")
    forcefield = app.ForceField(config.forcefield, config.water_model)

    # --- Add solvent ---
    if verbose:
        print(f"Adding solvent (padding={config.padding} nm, "
              f"ionic_strength={config.ionic_strength} M)")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    # Remove crystal water if present
    waters_to_delete = [
        r for r in modeller.topology.residues()
        if r.name in ("HOH", "WAT")
    ]
    if waters_to_delete:
        modeller.delete(waters_to_delete)
        if verbose:
            print(f"  Removed {len(waters_to_delete)} crystal waters")

    # Add missing hydrogens (PDB files typically lack them)
    if verbose:
        print("  Adding missing hydrogens")
    modeller.addHydrogens(forcefield)

    modeller.addSolvent(
        forcefield,
        padding=config.padding * unit.nanometers,
        ionicStrength=config.ionic_strength * unit.molar,
    )

    n_atoms_total = modeller.topology.getNumAtoms()
    if verbose:
        print(f"  System: {n_atoms_total} atoms (with solvent)")

    # --- Create system ---
    if verbose:
        print("Creating simulation system")

    constraints = {
        "HBonds": app.HBonds,
        "AllBonds": app.AllBonds,
        "HAngles": app.HAngles,
        "None": None,
    }[config.constraint_type]

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=config.nonbonded_cutoff * unit.nanometers,
        constraints=constraints,
    )

    # Add barostat for NPT
    system.addForce(openmm.MonteCarloBarostat(
        config.pressure * unit.atmospheres,
        config.temperature * unit.kelvin,
    ))

    # --- Integrator ---
    integrator = openmm.LangevinMiddleIntegrator(
        config.temperature * unit.kelvin,
        config.friction / unit.picoseconds,
        config.timestep * unit.picoseconds,
    )

    # --- Platform ---
    # Try CUDA/OpenCL first, fall back to CPU
    platform = _select_platform(verbose)

    simulation = app.Simulation(
        modeller.topology, system, integrator, platform
    )
    simulation.context.setPositions(modeller.positions)

    # --- Save solvated topology (needed to read DCD later) ---
    topology_path = output_dir / f"{stem}_solvated.pdb"
    with open(topology_path, "w") as f:
        app.PDBFile.writeFile(
            modeller.topology, modeller.positions, f
        )
    if verbose:
        print(f"  Saved solvated topology: {topology_path}")

    # --- Energy minimization ---
    if verbose:
        print("Minimizing energy...")
    e_before = simulation.context.getState(
        getEnergy=True
    ).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    simulation.minimizeEnergy(
        tolerance=config.minimize_tolerance * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=config.minimize_max_iter,
    )

    e_after = simulation.context.getState(
        getEnergy=True
    ).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    if verbose:
        print(f"  Energy: {e_before:.0f} -> {e_after:.0f} kJ/mol")

    # --- Equilibration (NVT-like via Langevin, with barostat = NPT) ---
    if verbose:
        equil_time = config.equilibration_steps * config.timestep / 1000  # ns
        print(f"Equilibrating ({equil_time:.2f} ns, "
              f"{config.equilibration_steps} steps)...")

    simulation.context.setVelocitiesToTemperature(
        config.temperature * unit.kelvin
    )
    simulation.step(config.equilibration_steps)

    if verbose:
        print("  Equilibration complete")

    # --- Production ---
    traj_path = output_dir / f"{stem}_traj.dcd"
    log_path = output_dir / f"{stem}_log.csv"
    forces_path = output_dir / f"{stem}_forces.npy"

    n_frames = config.production_steps // config.save_interval
    prod_time = config.production_steps * config.timestep / 1000  # ns

    if verbose:
        print(f"Production run ({prod_time:.2f} ns, "
              f"{config.production_steps} steps, {n_frames} frames)...")

    # Reset step counter for clean output
    simulation.currentStep = 0

    # Reporters
    simulation.reporters.append(
        app.DCDReporter(str(traj_path), config.save_interval)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            str(log_path),
            config.log_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            speed=True,
        )
    )

    force_reporter = None
    if save_forces:
        force_reporter = ForceReporter(forces_path, config.save_interval)
        simulation.reporters.append(force_reporter)

    # Run production
    simulation.step(config.production_steps)

    # Save forces
    if force_reporter is not None:
        force_reporter.save()
        if verbose:
            print(f"  Saved forces: {forces_path} "
                  f"({len(force_reporter._forces)} frames)")

    if verbose:
        print(f"  Saved trajectory: {traj_path} ({n_frames} frames)")
        print(f"  Saved log: {log_path}")
        print("Done.")

    return {
        "trajectory_path": traj_path,
        "forces_path": forces_path if save_forces else None,
        "topology_path": topology_path,
        "log_path": log_path,
        "n_frames": n_frames,
        "n_atoms": n_atoms_total,
        "elapsed_time_ns": prod_time,
    }


def _select_platform(verbose: bool = True) -> openmm.Platform:
    """Pick the fastest available OpenMM platform."""
    for name in ["CUDA", "OpenCL", "CPU"]:
        try:
            platform = openmm.Platform.getPlatformByName(name)
            if verbose:
                print(f"  Using platform: {name}")
            return platform
        except Exception:
            continue
    # Reference platform is always available
    if verbose:
        print("  Using platform: Reference (slow)")
    return openmm.Platform.getPlatformByName("Reference")
