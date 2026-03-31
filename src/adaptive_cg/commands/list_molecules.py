"""List downloaded molecular structures."""
from pathlib import Path

DATA_DIR = Path("data/structures")


def setup_parser(parser):
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Data directory (default: {DATA_DIR})",
    )


def execute(args):
    data_dir = args.data_dir

    if not data_dir.exists():
        print(f"No data directory found at {data_dir}/")
        print("Run 'acg fetch' first.")
        return 1

    pdb_files = sorted(data_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"No .pdb files in {data_dir}/")
        return 1

    print(f"Downloaded structures ({len(pdb_files)}):")
    for f in pdb_files:
        print(f"  {f.stem}")

    results_dir = Path("results")
    if results_dir.exists():
        evaluated = {p.stem for p in results_dir.glob("*.json")}
        n_eval = sum(1 for f in pdb_files if f.stem in evaluated)
        print(f"\nEvaluated: {n_eval}/{len(pdb_files)}")

    return 0
