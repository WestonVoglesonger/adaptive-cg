"""Command: extract CG distributions from AA trajectory."""
from __future__ import annotations

import argparse
from pathlib import Path


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "molecule",
        help="PDB ID (e.g. 1UBQ) — must have trajectory from `acg simulate`",
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
        "--nonbonded-cutoff", type=float, default=2.0,
        help="Max distance (nm) for non-bonded sampling (default: 2.0)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/extracted/<MOLECULE>)",
    )


def execute(args: argparse.Namespace) -> int:
    from adaptive_cg.core.extract import extract_distributions

    mol_id = args.molecule.upper()
    data_dir = Path("data")

    pdb_path = data_dir / "structures" / f"{mol_id}.pdb"
    traj_dir = data_dir / "trajectories" / mol_id

    if not pdb_path.exists():
        print(f"Error: PDB file not found: {pdb_path}")
        return 1
    if not traj_dir.exists():
        print(f"Error: Trajectory not found: {traj_dir}")
        print(f"Run `acg simulate {mol_id}` first")
        return 1

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "extracted" / mol_id

    print(f"=== CG Distribution Extraction: {mol_id} ===")
    print(f"Trajectory: {traj_dir}")
    print(f"Output: {output_dir}")
    print()

    result = extract_distributions(
        trajectory_dir=traj_dir,
        pdb_path=pdb_path,
        n_beads=args.n_beads,
        ratio=args.ratio,
        nonbonded_cutoff=args.nonbonded_cutoff,
        verbose=True,
    )

    result.save(output_dir)

    print()
    print("=== Summary ===")
    print(f"Frames processed: {result.n_frames}")
    print(f"Beads: {result.n_beads}")
    print(f"Bond types: {len(result.bond_distributions)}")
    print(f"Angle types: {len(result.angle_distributions)}")
    print(f"Non-bonded types: {len(result.nonbonded_distances)}")
    print(f"Saved to: {output_dir}")

    # Show distribution sizes
    print()
    print("Bond distributions:")
    for k, v in sorted(result.bond_distributions.items()):
        print(f"  {k}: {len(v)} samples")
    print()
    print("Bead class summary:")
    unique_keys = sorted(set(result.bead_class_keys))
    for k in unique_keys:
        count = result.bead_class_keys.count(k)
        print(f"  {k}: {count} beads")

    return 0
