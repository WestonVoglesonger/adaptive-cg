"""Command: derive transferable CG force field from extracted distributions."""
from __future__ import annotations

import argparse
from pathlib import Path


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--extracted-dir", type=str, default="data/extracted",
        help="Directory containing per-molecule extraction results",
    )
    parser.add_argument(
        "--temperature", type=float, default=300.0,
        help="Temperature in Kelvin (default: 300)",
    )
    parser.add_argument(
        "--output", type=str, default="data/forcefield/cg_forcefield.json",
        help="Output force field file path",
    )


def execute(args: argparse.Namespace) -> int:
    from adaptive_cg.core.parameterize import parameterize_forcefield

    extracted_root = Path(args.extracted_dir)
    if not extracted_root.exists():
        print(f"Error: Extracted data directory not found: {extracted_root}")
        print("Run `acg extract <MOLECULE>` first")
        return 1

    # Find all molecule directories with extraction results
    mol_dirs = sorted([
        d for d in extracted_root.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    ])

    if not mol_dirs:
        print(f"Error: No extraction results found in {extracted_root}")
        print("Run `acg extract <MOLECULE>` first")
        return 1

    mol_names = [d.name for d in mol_dirs]
    print(f"=== CG Force Field Parameterization ===")
    print(f"Molecules: {', '.join(mol_names)}")
    print(f"Temperature: {args.temperature} K")
    print()

    ff = parameterize_forcefield(
        extracted_dirs=mol_dirs,
        temperature=args.temperature,
        verbose=True,
    )

    output_path = Path(args.output)
    ff.save(output_path)

    print()
    print(f"=== Force field saved: {output_path} ===")
    print(f"Bond types: {len(ff.bond_params)}")
    print(f"Angle types: {len(ff.angle_params)}")
    print(f"Non-bonded types: {len(ff.nonbonded_params)}")
    print(f"Source molecules: {', '.join(ff.source_molecules)}")

    return 0
