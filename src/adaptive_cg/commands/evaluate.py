"""Evaluate CG mappings for a single molecule."""
from __future__ import annotations

import json
from pathlib import Path


def setup_parser(parser):
    parser.add_argument(
        "pdb_id",
        help="PDB ID to evaluate (e.g. 1UBQ)",
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
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--uniform-ratios",
        nargs="+",
        type=int,
        default=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        help="Uniform atom-to-bead ratios to test (default: 3-12)",
    )
    parser.add_argument(
        "--no-variable",
        action="store_true",
        help="Skip structure-aware variable mapping",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-region breakdown",
    )


def execute(args):
    from adaptive_cg.core.molecule import load_molecule
    from adaptive_cg.core.mapping import (
        eval_uniform_baselines,
        grid_search_variable,
    )

    pdb_id = args.pdb_id.upper()
    pdb_file = args.data_dir / f"{pdb_id}.pdb"

    if not pdb_file.exists():
        pdb_file = args.data_dir / f"{pdb_id.lower()}.pdb"
    if not pdb_file.exists():
        print(f"Error: PDB file not found: {args.data_dir / f'{pdb_id}.pdb'}")
        return 1

    mol = load_molecule(pdb_file)
    print(f"Loaded {mol}")

    # ------------------------------------------------------------------
    # Uniform baselines
    # ------------------------------------------------------------------
    print(f"\nUniform baselines (ratios: {args.uniform_ratios}):")
    uniform_results = eval_uniform_baselines(
        mol.positions, mol.masses, args.uniform_ratios,
    )

    for res in uniform_results:
        print(
            f"  ratio={res['ratio']:2d}  beads={res['n_beads']:4d}  "
            f"RMSE={res['rmse']:.6f}  MAE={res['mae']:.6f}  MRE={res['mre']:.4f}"
        )

    # Find best uniform by RMSE at each bead count
    best_uniform = min(uniform_results, key=lambda r: r["rmse"])
    print(
        f"\n  Best uniform: ratio={best_uniform['ratio']}, "
        f"RMSE={best_uniform['rmse']:.6f} ({best_uniform['n_beads']} beads)"
    )

    # ------------------------------------------------------------------
    # Variable (structure-aware) mapping via grid search
    # ------------------------------------------------------------------
    variable_result = None
    if not args.no_variable:
        target_beads = best_uniform["n_beads"]
        print(
            f"\nVariable mapping grid search "
            f"(target ~{target_beads} beads, regions: {mol.region_names}):"
        )

        variable_result = grid_search_variable(
            mol.positions,
            mol.masses,
            mol.region_labels,
            mol.region_names,
            target_beads=target_beads,
        )

        if variable_result["best_rmse"] < float("inf"):
            print(f"  Best variable RMSE: {variable_result['best_rmse']:.6f}")
            print(f"  Best ratios: {variable_result['best_ratios']}")
            print(f"  Beads: {variable_result['n_beads']}")

            improvement = best_uniform["rmse"] - variable_result["best_rmse"]
            pct = (improvement / best_uniform["rmse"]) * 100 if best_uniform["rmse"] > 0 else 0
            print(f"  Improvement over uniform: {improvement:.6f} nm ({pct:.1f}%)")

            if args.verbose and variable_result["all_results"]:
                print(f"\n  All variable results ({len(variable_result['all_results'])} combos):")
                sorted_results = sorted(variable_result["all_results"], key=lambda r: r["rmse"])
                for r in sorted_results[:10]:
                    print(
                        f"    ratios={r['ratios']}  beads={r['n_beads']}  "
                        f"RMSE={r['rmse']:.6f}"
                    )
                if len(sorted_results) > 10:
                    print(f"    ... ({len(sorted_results) - 10} more)")
        else:
            print("  No valid variable mappings found within bead tolerance.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = args.output_dir / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare serializable output (strip numpy distance matrices)
    uniform_out = []
    for res in uniform_results:
        uniform_out.append({
            "ratio": res["ratio"],
            "n_beads": res["n_beads"],
            "rmse": res["rmse"],
            "mae": res["mae"],
            "mre": res["mre"],
        })

    output = {
        "pdb_id": pdb_id,
        "mol_type": mol.mol_type,
        "n_atoms": mol.n_atoms,
        "region_names": mol.region_names,
        "uniform": uniform_out,
        "best_uniform_rmse": best_uniform["rmse"],
        "best_uniform_ratio": best_uniform["ratio"],
        "best_uniform_beads": best_uniform["n_beads"],
    }

    if variable_result and variable_result["best_rmse"] < float("inf"):
        output["variable"] = {
            "best_rmse": variable_result["best_rmse"],
            "best_ratios": variable_result["best_ratios"],
            "n_beads": variable_result["n_beads"],
            "n_combos_evaluated": len(variable_result["all_results"]),
        }
        output["improvement_nm"] = best_uniform["rmse"] - variable_result["best_rmse"]
        output["improvement_pct"] = (
            (output["improvement_nm"] / best_uniform["rmse"]) * 100
            if best_uniform["rmse"] > 0 else 0
        )

    out_file = out_dir / "evaluate_result.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_file}")

    return 0
