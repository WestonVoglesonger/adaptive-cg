"""Exp 3: Per-region RMSE breakdown for uniform vs variable CG mappings."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def setup_parser(parser):
    parser.add_argument(
        "pdb_ids",
        nargs="*",
        help="PDB IDs to analyze (default: all from sweep_summary.csv)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/structures"),
        help="Directory containing .pdb files",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory with sweep outputs (default: results/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/region_breakdown"),
        help="Output directory (default: results/region_breakdown/)",
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


def _load_pdb_ids_from_sweep(results_dir: Path) -> list[str]:
    """Read PDB IDs from an existing sweep_summary.csv."""
    csv_file = results_dir / "sweep_summary.csv"
    if not csv_file.exists():
        return []
    ids = []
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            ids.append(row["pdb_id"])
    return ids


def execute(args):
    import gc

    import numpy as np
    from scipy.spatial.distance import cdist

    from adaptive_cg.core.molecule import load_molecule
    from adaptive_cg.core.mapping import (
        eval_mapping_by_region,
        eval_uniform_baselines,
        generate_uniform_mapping,
        generate_variable_mapping,
        grid_search_variable,
    )

    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which PDB IDs to process
    if args.pdb_ids:
        pdb_ids = [p.upper() for p in args.pdb_ids]
    else:
        pdb_ids = _load_pdb_ids_from_sweep(args.results_dir)
        if not pdb_ids:
            print("No PDB IDs specified and no sweep_summary.csv found.")
            print("Run 'acg sweep' first, or pass PDB IDs as arguments.")
            return 1

    print(f"Exp 3: Per-region RMSE breakdown for {len(pdb_ids)} molecules")
    print()

    all_region_improvements: dict[str, list[dict]] = {}
    summary_rows = []
    n_success, n_fail = 0, 0

    for i, pdb_id in enumerate(pdb_ids, 1):
        pdb_file = data_dir / f"{pdb_id}.pdb"
        if not pdb_file.exists():
            pdb_file = data_dir / f"{pdb_id.lower()}.pdb"
        if not pdb_file.exists():
            print(f"[{i}/{len(pdb_ids)}] {pdb_id} ... FILE NOT FOUND")
            n_fail += 1
            continue

        print(f"[{i}/{len(pdb_ids)}] {pdb_id} ... ", end="", flush=True)

        try:
            mol = load_molecule(pdb_file)
        except Exception as e:
            print(f"LOAD FAILED: {e}")
            n_fail += 1
            continue

        if len(mol.region_names) < 2:
            print(f"SKIP (single region: {mol.region_names})")
            continue

        # -- Find best uniform mapping --
        uniform_results = eval_uniform_baselines(
            mol.positions, mol.masses, args.uniform_ratios,
        )
        best_uniform = min(uniform_results, key=lambda r: r["rmse"])
        best_uniform_ratio = best_uniform["ratio"]

        # Build the best uniform mapping for region breakdown
        uniform_mapping = generate_uniform_mapping(
            mol.n_atoms, best_uniform_ratio,
        )

        # -- Find best variable mapping --
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

        # -- Per-region breakdown for uniform --
        # Pre-compute AA distance matrix for reuse
        n_atoms = mol.positions.shape[0]
        aa_dmat = cdist(mol.positions, mol.positions) if n_atoms <= 1000 else None

        uniform_breakdown = eval_mapping_by_region(
            uniform_mapping, mol.positions, mol.masses,
            mol.region_labels, mol.region_names,
            aa_dmat=aa_dmat,
        )

        # -- Per-region breakdown for variable --
        variable_breakdown = None
        if has_variable:
            variable_breakdown = eval_mapping_by_region(
                variable_result["best_mapping"], mol.positions, mol.masses,
                mol.region_labels, mol.region_names,
                aa_dmat=aa_dmat,
            )

        # -- Build per-molecule result --
        mol_result = {
            "pdb_id": pdb_id,
            "mol_type": mol.mol_type,
            "n_atoms": mol.n_atoms,
            "region_names": mol.region_names,
            "region_counts": mol.region_counts,
            "uniform": {
                "ratio": best_uniform_ratio,
                "n_beads": best_uniform["n_beads"],
                "global_rmse": uniform_breakdown["global_rmse"],
                "per_region": uniform_breakdown["per_region"],
                "cross_region": uniform_breakdown["cross_region"],
            },
        }

        if variable_breakdown:
            mol_result["variable"] = {
                "ratios": variable_result["best_ratios"],
                "n_beads": variable_result["n_beads"],
                "global_rmse": variable_breakdown["global_rmse"],
                "per_region": variable_breakdown["per_region"],
                "cross_region": variable_breakdown["cross_region"],
            }

            # Compute per-region improvements
            mol_result["region_improvements"] = {}
            for rname in mol.region_names:
                uni_r = uniform_breakdown["per_region"].get(rname, {})
                var_r = variable_breakdown["per_region"].get(rname, {})
                uni_rmse = uni_r.get("rmse", 0.0)
                var_rmse = var_r.get("rmse", 0.0)
                if uni_rmse > 0 and uni_r.get("n_pairs", 0) > 0:
                    imp_nm = uni_rmse - var_rmse
                    imp_pct = imp_nm / uni_rmse * 100
                else:
                    imp_nm = 0.0
                    imp_pct = 0.0
                mol_result["region_improvements"][rname] = {
                    "improvement_nm": imp_nm,
                    "improvement_pct": imp_pct,
                }
                # Accumulate for aggregate table
                all_region_improvements.setdefault(rname, []).append({
                    "pdb_id": pdb_id,
                    "uniform_rmse": uni_rmse,
                    "variable_rmse": var_rmse,
                    "improvement_nm": imp_nm,
                    "improvement_pct": imp_pct,
                    "n_pairs": uni_r.get("n_pairs", 0),
                })

        # Save per-molecule JSON
        mol_out_dir = output_dir / pdb_id
        mol_out_dir.mkdir(parents=True, exist_ok=True)
        with open(mol_out_dir / "region_breakdown.json", "w") as f:
            json.dump(mol_result, f, indent=2)

        summary_rows.append(mol_result)
        n_success += 1

        # Print summary line
        regions_str = "  ".join(
            f"{rname}={uniform_breakdown['per_region'].get(rname, {}).get('rmse', 0.0):.4f}"
            for rname in mol.region_names
        )
        print(f"{mol.n_atoms} atoms, uni_global={uniform_breakdown['global_rmse']:.4f}, {regions_str}")

        del mol, uniform_results, variable_result, aa_dmat
        gc.collect()

    # ------------------------------------------------------------------
    # Aggregate table: mean improvement by region type
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"Exp 3 complete: {n_success} succeeded, {n_fail} failed")
    print("=" * 70)
    print()

    if all_region_improvements:
        print("Per-region mean improvement (variable vs uniform):")
        print(f"  {'Region':<15s} {'Mean Imp%':>10s} {'Mean Imp nm':>12s} "
              f"{'Uni RMSE':>10s} {'Var RMSE':>10s} {'N molecules':>12s}")
        print("-" * 70)

        aggregate = {}
        for rname, entries in sorted(all_region_improvements.items()):
            valid = [e for e in entries if e["n_pairs"] > 0]
            if not valid:
                continue
            import numpy as np
            imp_pcts = np.array([e["improvement_pct"] for e in valid])
            imp_nms = np.array([e["improvement_nm"] for e in valid])
            uni_rmses = np.array([e["uniform_rmse"] for e in valid])
            var_rmses = np.array([e["variable_rmse"] for e in valid])

            aggregate[rname] = {
                "mean_improvement_pct": float(imp_pcts.mean()),
                "mean_improvement_nm": float(imp_nms.mean()),
                "mean_uniform_rmse": float(uni_rmses.mean()),
                "mean_variable_rmse": float(var_rmses.mean()),
                "n_molecules": len(valid),
            }

            print(
                f"  {rname:<15s} {imp_pcts.mean():>+9.1f}% {imp_nms.mean():>11.6f} "
                f"{uni_rmses.mean():>10.6f} {var_rmses.mean():>10.6f} {len(valid):>12d}"
            )

        # Save aggregate
        agg_file = output_dir / "region_aggregate.json"
        with open(agg_file, "w") as f:
            json.dump(aggregate, f, indent=2)
        print(f"\n  Aggregate JSON: {agg_file}")

    # Save full summary
    summary_file = output_dir / "region_breakdown_summary.json"
    serializable = []
    for row in summary_rows:
        serializable.append({
            "pdb_id": row["pdb_id"],
            "mol_type": row["mol_type"],
            "n_atoms": row["n_atoms"],
            "region_names": row["region_names"],
            "uniform_global_rmse": row["uniform"]["global_rmse"],
            "uniform_per_region": row["uniform"]["per_region"],
            "variable_global_rmse": row.get("variable", {}).get("global_rmse"),
            "variable_per_region": row.get("variable", {}).get("per_region"),
            "region_improvements": row.get("region_improvements"),
        })
    with open(summary_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Summary JSON: {summary_file}")

    return 0
