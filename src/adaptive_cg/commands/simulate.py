"""Command: run all-atom MD simulation for CG force field parameterization."""
from __future__ import annotations

import argparse
from pathlib import Path


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "molecule",
        help="PDB ID (e.g. 1UBQ) — must already be fetched via `acg fetch`",
    )
    parser.add_argument(
        "--steps", type=int, default=500_000,
        help="Production steps (default: 500000 = 1 ns at 2 fs timestep)",
    )
    parser.add_argument(
        "--equilibration", type=int, default=25_000,
        help="Equilibration steps (default: 25000 = 50 ps)",
    )
    parser.add_argument(
        "--save-interval", type=int, default=1000,
        help="Save trajectory every N steps (default: 1000 = 2 ps)",
    )
    parser.add_argument(
        "--timestep", type=float, default=0.002,
        help="Integration timestep in ps (default: 0.002 = 2 fs)",
    )
    parser.add_argument(
        "--temperature", type=float, default=300.0,
        help="Temperature in Kelvin (default: 300)",
    )
    parser.add_argument(
        "--no-forces", action="store_true",
        help="Skip saving per-frame forces (saves disk space)",
    )
    parser.add_argument(
        "--padding", type=float, default=1.0,
        help="Water box padding in nm (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/trajectories/<MOLECULE>)",
    )


def execute(args: argparse.Namespace) -> int:
    from adaptive_cg.core.simulation import SimulationConfig, run_aa_simulation

    mol_id = args.molecule.upper()
    data_dir = Path("data")
    pdb_path = data_dir / "structures" / f"{mol_id}.pdb"

    if not pdb_path.exists():
        print(f"Error: PDB file not found: {pdb_path}")
        print(f"Run `acg fetch` first to download {mol_id}")
        return 1

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "trajectories" / mol_id

    config = SimulationConfig(
        production_steps=args.steps,
        equilibration_steps=args.equilibration,
        save_interval=args.save_interval,
        timestep=args.timestep,
        temperature=args.temperature,
        padding=args.padding,
    )

    prod_time = config.production_steps * config.timestep / 1000
    print(f"=== AA MD Simulation: {mol_id} ===")
    print(f"Production: {prod_time:.2f} ns ({config.production_steps} steps)")
    print(f"Timestep: {config.timestep * 1000:.1f} fs")
    print(f"Temperature: {config.temperature} K")
    print(f"Save interval: {config.save_interval} steps "
          f"({config.save_interval * config.timestep:.1f} ps)")
    print(f"Output: {output_dir}")
    print()

    result = run_aa_simulation(
        pdb_path=pdb_path,
        output_dir=output_dir,
        config=config,
        save_forces=not args.no_forces,
        verbose=True,
    )

    print()
    print("=== Summary ===")
    print(f"Trajectory: {result['trajectory_path']}")
    print(f"Topology:   {result['topology_path']}")
    if result["forces_path"]:
        print(f"Forces:     {result['forces_path']}")
    print(f"Log:        {result['log_path']}")
    print(f"Frames:     {result['n_frames']}")
    print(f"Atoms:      {result['n_atoms']} (with solvent)")
    print(f"Time:       {result['elapsed_time_ns']:.2f} ns")

    return 0
