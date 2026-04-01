"""Command: run coarse-grained molecular dynamics simulation."""
from __future__ import annotations

import argparse
from pathlib import Path


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "molecule",
        help="PDB ID (e.g. 1UBQ) — must already be fetched via `acg fetch`",
    )
    parser.add_argument(
        "--steps", type=int, default=10000,
        help="Number of integration steps (default: 10000)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.01,
        help="Timestep in ps (default: 0.01 = 10 fs)",
    )
    parser.add_argument(
        "--temperature", type=float, default=300.0,
        help="Temperature in Kelvin (default: 300)",
    )
    parser.add_argument(
        "--friction", type=float, default=1.0,
        help="Langevin friction coefficient in 1/ps (default: 1.0)",
    )
    parser.add_argument(
        "--thermostat", choices=["langevin", "nve"], default="langevin",
        help="Thermostat type (default: langevin)",
    )
    parser.add_argument(
        "--n-beads", type=int, default=None,
        help="Number of CG beads (default: n_atoms // ratio)",
    )
    parser.add_argument(
        "--ratio", type=int, default=4,
        help="Default atom-to-bead ratio if --n-beads not set (default: 4)",
    )
    parser.add_argument(
        "--forcefield", type=str, default="data/forcefield/cg_forcefield.json",
        help="Path to CG force field JSON",
    )
    # Force field scaling
    parser.add_argument(
        "--bond-scale", type=float, default=1.0,
        help="Scale bond force constants (default: 1.0)",
    )
    parser.add_argument(
        "--angle-scale", type=float, default=1.0,
        help="Scale angle force constants (default: 1.0)",
    )
    parser.add_argument(
        "--dihedral-scale", type=float, default=1.0,
        help="Scale dihedral force constants (default: 1.0)",
    )
    parser.add_argument(
        "--structure-bias", choices=["none", "elastic", "go"], default="none",
        help="Structure bias to prevent unfolding (default: none)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=100,
        help="Log energies every N steps (default: 100)",
    )
    parser.add_argument(
        "--save-interval", type=int, default=1000,
        help="Save trajectory frame every N steps (default: 1000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/cg_trajectories/<MOLECULE>)",
    )


def execute(args: argparse.Namespace) -> int:
    from adaptive_cg.core.engine import setup_cg_system, run_cg_simulation

    mol_id = args.molecule.upper()
    data_dir = Path("data")
    pdb_path = data_dir / "structures" / f"{mol_id}.pdb"
    ff_path = Path(args.forcefield)

    if not pdb_path.exists():
        print(f"Error: PDB file not found: {pdb_path}")
        print(f"Run `acg fetch` first to download {mol_id}")
        return 1

    if not ff_path.exists():
        print(f"Error: Force field not found: {ff_path}")
        print("Run `acg parameterize` first")
        return 1

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "cg_trajectories" / mol_id

    output_dir.mkdir(parents=True, exist_ok=True)

    total_time = args.steps * args.dt
    print(f"=== CG MD Simulation: {mol_id} ===")
    print(f"Steps: {args.steps}, dt={args.dt} ps, total={total_time:.2f} ps")
    print(f"Temperature: {args.temperature} K")
    print(f"Thermostat: {args.thermostat}")
    print(f"Force field: {ff_path}")
    print(f"Output: {output_dir}")
    print()

    # Setup system
    system = setup_cg_system(
        pdb_path=pdb_path,
        ff_path=ff_path,
        n_beads=args.n_beads,
        ratio=args.ratio,
        temperature=args.temperature,
        bond_scale=args.bond_scale,
        angle_scale=args.angle_scale,
        dihedral_scale=args.dihedral_scale,
        structure_bias=args.structure_bias,
        verbose=True,
    )

    print()

    # Run simulation
    traj_path = output_dir / f"{mol_id}_cg_traj.npy"
    log = run_cg_simulation(
        system=system,
        n_steps=args.steps,
        dt=args.dt,
        temperature=args.temperature,
        friction=args.friction,
        thermostat=args.thermostat,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        trajectory_path=traj_path,
        verbose=True,
    )

    # Save log
    log_path = output_dir / f"{mol_id}_cg_log.csv"
    log.save(log_path)

    print()
    print("=== Summary ===")
    print(f"Trajectory: {traj_path}")
    print(f"Log: {log_path}")
    print(f"Beads: {system.n_beads}")
    if log.temperature:
        print(f"Final temperature: {log.temperature[-1]:.1f} K")
        print(f"Final total energy: {log.total_energy[-1]:.1f} kJ/mol")

    return 0
