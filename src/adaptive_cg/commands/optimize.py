"""PyTorch differentiable CG optimization for a single molecule."""
from pathlib import Path


def setup_parser(parser):
    parser.add_argument(
        "pdb_id",
        help="PDB ID to optimize (e.g. 1UBQ)",
    )
    parser.add_argument(
        "--beads",
        type=int,
        required=True,
        help="Target number of beads",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/structures"),
        help="Directory containing .pdb files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Optimization epochs (default: 2000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=200,
        help="Skip molecules with more heavy atoms (default: 200)",
    )


def execute(args):
    import json
    from adaptive_cg.core.molecule import load_molecule
    from adaptive_cg.core.optimizer import DifferentiableCGOptimizer

    pdb_id = args.pdb_id.upper()
    pdb_file = args.data_dir / f"{pdb_id}.pdb"

    if not pdb_file.exists():
        # Try lowercase
        pdb_file = args.data_dir / f"{pdb_id.lower()}.pdb"
    if not pdb_file.exists():
        print(f"Error: PDB file not found: {args.data_dir / f'{pdb_id}.pdb'}")
        return 1

    mol = load_molecule(pdb_file)
    print(f"Loaded {mol}")

    if mol.n_atoms > args.max_atoms:
        print(
            f"Skipping: {mol.n_atoms} atoms exceeds --max-atoms={args.max_atoms}. "
            f"Differentiable optimizer is O(N*B) and impractical for large molecules."
        )
        return 1

    print(f"\nOptimizing {pdb_id}: {mol.n_atoms} atoms -> {args.beads} beads")
    print(f"  epochs={args.epochs}  lr={args.lr}")

    opt = DifferentiableCGOptimizer(
        positions=mol.positions,
        masses=mol.masses,
        n_beads=args.beads,
        epochs=args.epochs,
        lr=args.lr,
    )
    result = opt.optimize()

    print(f"\nResult: {result['n_beads_used']} beads used, RMSE={result['rmse']:.6f} nm")

    # Save results
    out_dir = args.output_dir / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "optimize_result.json"
    serializable = {
        "pdb_id": pdb_id,
        "n_atoms": mol.n_atoms,
        "n_beads_target": args.beads,
        "n_beads_used": result["n_beads_used"],
        "rmse": result["rmse"],
        "final_temperature": result["final_temperature"],
        "mapping": result["mapping"],
    }
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved to {out_file}")

    return 0
