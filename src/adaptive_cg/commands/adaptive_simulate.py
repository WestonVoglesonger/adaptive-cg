"""Command: run adaptive resolution CG MD simulation."""
from __future__ import annotations

import argparse
from pathlib import Path


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "molecule",
        help="PDB ID (e.g. 1UBQ) — must already be fetched via `acg fetch`",
    )
    parser.add_argument(
        "--steps", type=int, default=100000,
        help="Number of integration steps (default: 100000)",
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
        help="Total CG beads (fixed, redistributed by activity)",
    )
    parser.add_argument(
        "--ratio", type=int, default=4,
        help="Default atom-to-bead ratio if --n-beads not set (default: 4)",
    )
    parser.add_argument(
        "--forcefield", type=str, default="data/forcefield/cg_forcefield.json",
        help="Path to CG force field JSON",
    )
    # Adaptive parameters
    parser.add_argument(
        "--n-regions", type=int, default=5,
        help="Number of chain regions to monitor (default: 5)",
    )
    parser.add_argument(
        "--remap-interval", type=int, default=5000,
        help="Steps between remap checks (default: 5000)",
    )
    parser.add_argument(
        "--remap-threshold", type=int, default=3,
        help="Min bead difference to trigger remap (default: 3)",
    )
    parser.add_argument(
        "--activity-weight", type=float, default=0.5,
        help="Activity vs size weight for allocation, 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--monitor-interval", type=int, default=100,
        help="Steps between activity recordings (default: 100)",
    )
    parser.add_argument(
        "--monitor-window", type=int, default=50,
        help="Sliding window size for RMSF (default: 50)",
    )
    # Force field scaling
    parser.add_argument(
        "--dihedral-scale", type=float, default=1.0,
        help="Scale dihedral force constants (default: 1.0)",
    )
    parser.add_argument(
        "--structure-bias", choices=["none", "elastic", "go"], default="none",
        help="Structure bias to prevent unfolding (default: none)",
    )
    # Logging
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
        help="Output directory (default: data/adaptive_trajectories/<MOL>)",
    )


def execute(args: argparse.Namespace) -> int:
    from adaptive_cg.core.adaptive import run_adaptive_simulation

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
        output_dir = data_dir / "adaptive_trajectories" / mol_id

    output_dir.mkdir(parents=True, exist_ok=True)

    total_time = args.steps * args.dt
    print(f"=== Adaptive CG MD: {mol_id} ===")
    print(f"Steps: {args.steps}, dt={args.dt} ps, total={total_time:.2f} ps")
    print(f"Temperature: {args.temperature} K, thermostat: {args.thermostat}")
    print(f"Regions: {args.n_regions}, remap every {args.remap_interval} steps")
    print(f"Activity weight: {args.activity_weight}")
    print(f"Force field: {ff_path}")
    print(f"Output: {output_dir}")
    print()

    traj_path = output_dir / f"{mol_id}_adaptive_traj.npz"

    log = run_adaptive_simulation(
        pdb_path=pdb_path,
        ff_path=ff_path,
        n_beads=args.n_beads,
        ratio=args.ratio,
        n_steps=args.steps,
        dt=args.dt,
        temperature=args.temperature,
        friction=args.friction,
        thermostat=args.thermostat,
        n_regions=args.n_regions,
        monitor_interval=args.monitor_interval,
        remap_check_interval=args.remap_interval,
        remap_threshold=args.remap_threshold,
        activity_weight=args.activity_weight,
        monitor_window=args.monitor_window,
        dihedral_scale=args.dihedral_scale,
        structure_bias=args.structure_bias,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        trajectory_path=traj_path,
        verbose=True,
    )

    # Save logs
    log_path = output_dir / f"{mol_id}_adaptive_log.csv"
    log.save(log_path)

    events_path = output_dir / f"{mol_id}_adaptive_events.json"
    log.save_adaptive(events_path)

    print()
    print("=== Summary ===")
    print(f"Trajectory: {traj_path}")
    print(f"Energy log: {log_path}")
    print(f"Adaptive events: {events_path}")
    print(f"Total remaps: {len(log.remap_steps)}")
    if log.remap_steps:
        for i, step in enumerate(log.remap_steps):
            print(f"  Step {step}: "
                  f"{log.remap_old_beads[i]} -> {log.remap_new_beads[i]} beads")
    if log.temperature:
        print(f"Final temperature: {log.temperature[-1]:.1f} K")
        print(f"Final total energy: {log.total_energy[-1]:.1f} kJ/mol")

    return 0
