"""Run evaluation across all downloaded molecules (Phase 1)."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def setup_parser(parser):
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
        "--skip-optimize",
        action="store_true",
        help="Skip PyTorch optimization (grid search only)",
    )
    parser.add_argument(
        "--max-optimize-atoms",
        type=int,
        default=200,
        help="Only run PyTorch optimization on molecules with <= N heavy atoms",
    )
    parser.add_argument(
        "--uniform-ratios",
        nargs="+",
        type=int,
        default=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        help="Uniform ratios to test (default: 3-12)",
    )
    parser.add_argument(
        "--grid-ratio-range",
        nargs=2,
        type=int,
        default=[3, 10],
        metavar=("LO", "HI"),
        help="Variable grid search ratio range (default: 3 10)",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Skip molecules with more than N heavy atoms",
    )


def execute(args):
    import gc

    from adaptive_cg.core.molecule import load_molecule
    from adaptive_cg.core.mapping import (
        eval_uniform_baselines,
        grid_search_variable,
    )

    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"No data directory found at {data_dir}/")
        print("Run 'acg fetch' first.")
        return 1

    pdb_files = sorted(data_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"No .pdb files in {data_dir}/")
        return 1

    # Sort smallest files first so small molecules complete before big ones.
    pdb_files.sort(key=lambda p: p.stat().st_size)

    print(f"Phase 1: sweep across {len(pdb_files)} molecules (smallest first)")
    print(f"  Uniform ratios: {args.uniform_ratios}")
    print(f"  Grid ratio range: {args.grid_ratio_range[0]}-{args.grid_ratio_range[1]}")
    print(f"  PyTorch optimize: {'skip' if args.skip_optimize else f'<= {args.max_optimize_atoms} atoms'}")
    if args.max_atoms:
        print(f"  Max atoms: {args.max_atoms}")
    print()

    summary_rows = []
    n_success, n_fail = 0, 0

    # Prepare CSV for incremental writes (survives kill/crash).
    csv_file = output_dir / "sweep_summary.csv"
    csv_fieldnames = [
        "pdb_id", "mol_type", "n_atoms", "n_regions", "region_names",
        "best_uniform_ratio", "best_uniform_beads", "best_uniform_rmse",
        "best_uniform_mae", "variable_rmse", "variable_ratios",
        "variable_beads", "improvement_nm", "improvement_pct",
        "optimize_rmse", "optimize_beads",
    ]
    csv_fh = open(csv_file, "w", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=csv_fieldnames)
    csv_writer.writeheader()
    csv_fh.flush()

    for i, pdb_file in enumerate(pdb_files, 1):
        pdb_id = pdb_file.stem.upper()
        print(f"[{i}/{len(pdb_files)}] {pdb_id} ... ", end="", flush=True)

        try:
            mol = load_molecule(pdb_file)
        except Exception as e:
            print(f"LOAD FAILED: {e}")
            n_fail += 1
            continue

        if args.max_atoms and mol.n_atoms > args.max_atoms:
            print(f"SKIP ({mol.n_atoms} atoms > {args.max_atoms})")
            continue

        # -- Uniform baselines --
        uniform_results = eval_uniform_baselines(
            mol.positions, mol.masses, args.uniform_ratios,
        )
        best_uniform = min(uniform_results, key=lambda r: r["rmse"])

        # -- Variable mapping --
        # Scale tolerance with molecule size so larger molecules still find configs.
        target_beads = best_uniform["n_beads"]
        tol = max(3, target_beads // 10)
        variable_result = grid_search_variable(
            mol.positions,
            mol.masses,
            mol.region_labels,
            mol.region_names,
            ratio_range=tuple(args.grid_ratio_range),
            target_beads=target_beads,
            tolerance=tol,
        )

        has_variable = variable_result["best_rmse"] < float("inf")

        # -- Optional PyTorch optimization --
        opt_result = None
        if (
            not args.skip_optimize
            and mol.n_atoms <= args.max_optimize_atoms
            and target_beads >= 2
        ):
            try:
                from adaptive_cg.core.optimizer import DifferentiableCGOptimizer
                opt = DifferentiableCGOptimizer(
                    positions=mol.positions,
                    masses=mol.masses,
                    n_beads=target_beads,
                    epochs=1000,
                )
                opt_result = opt.optimize()
            except ImportError:
                pass  # PyTorch not installed
            except Exception as e:
                print(f"(optimize failed: {e}) ", end="")

        # -- Build row --
        row = {
            "pdb_id": pdb_id,
            "mol_type": mol.mol_type,
            "n_atoms": mol.n_atoms,
            "n_regions": len(mol.region_names),
            "region_names": ",".join(mol.region_names),
            "best_uniform_ratio": best_uniform["ratio"],
            "best_uniform_beads": best_uniform["n_beads"],
            "best_uniform_rmse": best_uniform["rmse"],
            "best_uniform_mae": best_uniform["mae"],
            "variable_rmse": variable_result["best_rmse"] if has_variable else None,
            "variable_ratios": json.dumps(variable_result["best_ratios"]) if has_variable else None,
            "variable_beads": variable_result["n_beads"] if has_variable else None,
            "improvement_nm": (
                best_uniform["rmse"] - variable_result["best_rmse"]
                if has_variable else None
            ),
            "improvement_pct": (
                (best_uniform["rmse"] - variable_result["best_rmse"])
                / best_uniform["rmse"] * 100
                if has_variable and best_uniform["rmse"] > 0 else None
            ),
            "optimize_rmse": opt_result["rmse"] if opt_result else None,
            "optimize_beads": opt_result["n_beads_used"] if opt_result else None,
        }
        summary_rows.append(row)
        n_success += 1

        # Print summary line
        var_str = f"var={variable_result['best_rmse']:.4f}" if has_variable else "var=N/A"
        opt_str = f"opt={opt_result['rmse']:.4f}" if opt_result else ""
        print(
            f"{mol.n_atoms} atoms, {mol.mol_type}, "
            f"uni={best_uniform['rmse']:.4f}, {var_str} "
            f"{opt_str}"
        )

        # Save per-molecule JSON
        mol_dir = output_dir / pdb_id
        mol_dir.mkdir(parents=True, exist_ok=True)

        uniform_out = [
            {
                "ratio": r["ratio"],
                "n_beads": r["n_beads"],
                "rmse": r["rmse"],
                "mae": r["mae"],
                "mre": r["mre"],
            }
            for r in uniform_results
        ]

        mol_result = {
            "pdb_id": pdb_id,
            "mol_type": mol.mol_type,
            "n_atoms": mol.n_atoms,
            "region_names": mol.region_names,
            "uniform": uniform_out,
            "best_uniform_rmse": best_uniform["rmse"],
            "best_uniform_ratio": best_uniform["ratio"],
        }

        if has_variable:
            mol_result["variable"] = {
                "best_rmse": variable_result["best_rmse"],
                "best_ratios": variable_result["best_ratios"],
                "n_beads": variable_result["n_beads"],
                "n_combos_evaluated": len(variable_result["all_results"]),
            }
            mol_result["improvement_nm"] = row["improvement_nm"]
            mol_result["improvement_pct"] = row["improvement_pct"]

        if opt_result:
            mol_result["optimize"] = {
                "rmse": opt_result["rmse"],
                "n_beads_used": opt_result["n_beads_used"],
                "mapping": opt_result["mapping"],
            }

        with open(mol_dir / "evaluate_result.json", "w") as f:
            json.dump(mol_result, f, indent=2)

        # Write row to CSV incrementally so results survive crashes.
        csv_writer.writerow(row)
        csv_fh.flush()

        # Free memory between molecules to avoid OOM on constrained machines.
        del mol, uniform_results, variable_result, opt_result, mol_result
        gc.collect()

    csv_fh.close()

    # ------------------------------------------------------------------
    # Write summary JSON
    # ------------------------------------------------------------------
    if summary_rows:
        json_file = output_dir / "sweep_summary.json"
        with open(json_file, "w") as f:
            json.dump(summary_rows, f, indent=2, default=str)
        print(f"\nSummary CSV: {csv_file}")
        print(f"Summary JSON: {json_file}")

    # ------------------------------------------------------------------
    # Print aggregate stats
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 1 complete: {n_success} succeeded, {n_fail} failed")

    if summary_rows:
        improvements = [
            r["improvement_pct"] for r in summary_rows
            if r["improvement_pct"] is not None
        ]
        if improvements:
            import numpy as np
            arr = np.array(improvements)
            print(f"\nVariable vs Uniform improvement:")
            print(f"  Mean:   {arr.mean():.1f}%")
            print(f"  Median: {np.median(arr):.1f}%")
            print(f"  Min:    {arr.min():.1f}%")
            print(f"  Max:    {arr.max():.1f}%")
            print(f"  >0%:    {(arr > 0).sum()}/{len(arr)} molecules")

    return 0
