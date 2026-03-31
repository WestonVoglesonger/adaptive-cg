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
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Skip molecules with more than N heavy atoms",
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
    from adaptive_cg.core.strategies import kmeans_mapping, hierarchical_mapping

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

        if args.max_atoms and mol.n_atoms > args.max_atoms:
            print(f"SKIP ({mol.n_atoms} atoms > {args.max_atoms})")
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

        # -- K-means and hierarchical mappings --
        target_beads = best_uniform["n_beads"]
        try:
            km_mapping = kmeans_mapping(mol.positions, mol.masses, target_beads)
            km_breakdown = eval_mapping_by_region(
                km_mapping, mol.positions, mol.masses,
                mol.region_labels, mol.region_names,
                aa_dmat=aa_dmat,
            )
        except Exception:
            km_mapping = None
            km_breakdown = None

        try:
            hi_mapping = hierarchical_mapping(mol.positions, mol.masses, target_beads)
            hi_breakdown = eval_mapping_by_region(
                hi_mapping, mol.positions, mol.masses,
                mol.region_labels, mol.region_names,
                aa_dmat=aa_dmat,
            )
        except Exception:
            hi_mapping = None
            hi_breakdown = None

        # -- Analyze bead distribution across regions for each strategy --
        def _bead_region_distribution(mapping, region_labels, region_names):
            """Count how many beads are assigned to each region (by majority vote)."""
            counts = {rn: 0 for rn in region_names}
            for group in mapping:
                labels_in_bead = region_labels[group]
                majority = int(np.bincount(labels_in_bead).argmax())
                if majority < len(region_names):
                    counts[region_names[majority]] += 1
            return counts

        uniform_bead_dist = _bead_region_distribution(
            uniform_mapping, mol.region_labels, mol.region_names)
        km_bead_dist = _bead_region_distribution(
            km_mapping, mol.region_labels, mol.region_names) if km_mapping else None
        hi_bead_dist = _bead_region_distribution(
            hi_mapping, mol.region_labels, mol.region_names) if hi_mapping else None

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
                "bead_distribution": uniform_bead_dist,
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

        if km_breakdown:
            mol_result["kmeans"] = {
                "n_beads": len(km_mapping),
                "global_rmse": km_breakdown["global_rmse"],
                "per_region": km_breakdown["per_region"],
                "cross_region": km_breakdown["cross_region"],
                "bead_distribution": km_bead_dist,
            }

        if hi_breakdown:
            mol_result["hierarchical"] = {
                "n_beads": len(hi_mapping),
                "global_rmse": hi_breakdown["global_rmse"],
                "per_region": hi_breakdown["per_region"],
                "cross_region": hi_breakdown["cross_region"],
                "bead_distribution": hi_bead_dist,
            }

            # Compute per-region improvements for all strategies vs uniform
            mol_result["region_improvements"] = {}
            for rname in mol.region_names:
                uni_r = uniform_breakdown["per_region"].get(rname, {})
                uni_rmse = uni_r.get("rmse", 0.0)
                n_pairs = uni_r.get("n_pairs", 0)

                entry = {"uniform_rmse": uni_rmse, "n_pairs": n_pairs}

                for strat_name, strat_breakdown in [
                    ("grid_search", variable_breakdown),
                    ("kmeans", km_breakdown),
                    ("hierarchical", hi_breakdown),
                ]:
                    if strat_breakdown is None:
                        continue
                    s_r = strat_breakdown["per_region"].get(rname, {})
                    s_rmse = s_r.get("rmse", 0.0)
                    if uni_rmse > 0 and n_pairs > 0:
                        imp_nm = uni_rmse - s_rmse
                        imp_pct = imp_nm / uni_rmse * 100
                    else:
                        imp_nm = 0.0
                        imp_pct = 0.0
                    entry[f"{strat_name}_rmse"] = s_rmse
                    entry[f"{strat_name}_improvement_pct"] = imp_pct

                mol_result["region_improvements"][rname] = entry

                # Accumulate for aggregate table
                all_region_improvements.setdefault(rname, []).append({
                    "pdb_id": pdb_id,
                    "uniform_rmse": uni_rmse,
                    "variable_rmse": variable_breakdown["per_region"].get(rname, {}).get("rmse", 0.0) if variable_breakdown else None,
                    "kmeans_rmse": km_breakdown["per_region"].get(rname, {}).get("rmse", 0.0) if km_breakdown else None,
                    "hierarchical_rmse": hi_breakdown["per_region"].get(rname, {}).get("rmse", 0.0) if hi_breakdown else None,
                    "n_pairs": n_pairs,
                })

        # Save per-molecule JSON
        mol_out_dir = output_dir / pdb_id
        mol_out_dir.mkdir(parents=True, exist_ok=True)
        with open(mol_out_dir / "region_breakdown.json", "w") as f:
            json.dump(mol_result, f, indent=2)

        summary_rows.append(mol_result)
        n_success += 1

        # Print summary line
        km_str = f"km={km_breakdown['global_rmse']:.4f}" if km_breakdown else "km=N/A"
        hi_str = f"hi={hi_breakdown['global_rmse']:.4f}" if hi_breakdown else "hi=N/A"
        print(
            f"{mol.n_atoms} atoms, uni={uniform_breakdown['global_rmse']:.4f}, "
            f"{km_str}, {hi_str}"
        )

        del mol, uniform_results, variable_result, aa_dmat, km_mapping, hi_mapping
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
        print("Per-region mean RMSE by strategy:")
        print(f"  {'Region':<15s} {'Uniform':>10s} {'GridSrch':>10s} "
              f"{'K-means':>10s} {'Hierarch':>10s} {'N':>4s}")
        print("-" * 70)

        aggregate = {}
        for rname, entries in sorted(all_region_improvements.items()):
            valid = [e for e in entries if e["n_pairs"] > 0]
            if not valid:
                continue

            uni_rmses = np.array([e["uniform_rmse"] for e in valid])
            gs_rmses = np.array([e["variable_rmse"] for e in valid if e.get("variable_rmse") is not None])
            km_rmses = np.array([e["kmeans_rmse"] for e in valid if e.get("kmeans_rmse") is not None])
            hi_rmses = np.array([e["hierarchical_rmse"] for e in valid if e.get("hierarchical_rmse") is not None])

            aggregate[rname] = {
                "mean_uniform_rmse": float(uni_rmses.mean()),
                "mean_grid_search_rmse": float(gs_rmses.mean()) if len(gs_rmses) else None,
                "mean_kmeans_rmse": float(km_rmses.mean()) if len(km_rmses) else None,
                "mean_hierarchical_rmse": float(hi_rmses.mean()) if len(hi_rmses) else None,
                "n_molecules": len(valid),
            }

            gs_str = f"{gs_rmses.mean():>10.6f}" if len(gs_rmses) else f"{'N/A':>10s}"
            km_str = f"{km_rmses.mean():>10.6f}" if len(km_rmses) else f"{'N/A':>10s}"
            hi_str = f"{hi_rmses.mean():>10.6f}" if len(hi_rmses) else f"{'N/A':>10s}"
            print(
                f"  {rname:<15s} {uni_rmses.mean():>10.6f} {gs_str} "
                f"{km_str} {hi_str} {len(valid):>4d}"
            )

        # Bead distribution analysis
        print()
        print("Mean bead allocation by region (atoms per bead = effective ratio):")
        print(f"  {'Region':<15s} {'Atom%':>7s} {'Uni beads%':>11s} "
              f"{'KM beads%':>11s} {'Hi beads%':>11s}")
        print("-" * 65)

        # Collect bead distribution data across molecules
        for rname in sorted(set(r for row in summary_rows for r in row["region_names"])):
            atom_pcts = []
            uni_bead_pcts = []
            km_bead_pcts = []
            hi_bead_pcts = []
            for row in summary_rows:
                if rname not in row["region_names"]:
                    continue
                total_atoms = row["n_atoms"]
                region_atoms = row["region_counts"].get(rname, 0)
                atom_pcts.append(region_atoms / total_atoms * 100)

                uni_total = sum(row["uniform"]["bead_distribution"].values())
                if uni_total > 0:
                    uni_bead_pcts.append(
                        row["uniform"]["bead_distribution"].get(rname, 0) / uni_total * 100)

                if "kmeans" in row and row["kmeans"].get("bead_distribution"):
                    km_total = sum(row["kmeans"]["bead_distribution"].values())
                    if km_total > 0:
                        km_bead_pcts.append(
                            row["kmeans"]["bead_distribution"].get(rname, 0) / km_total * 100)

                if "hierarchical" in row and row["hierarchical"].get("bead_distribution"):
                    hi_total = sum(row["hierarchical"]["bead_distribution"].values())
                    if hi_total > 0:
                        hi_bead_pcts.append(
                            row["hierarchical"]["bead_distribution"].get(rname, 0) / hi_total * 100)

            if not atom_pcts:
                continue
            a_str = f"{np.mean(atom_pcts):>6.1f}%"
            u_str = f"{np.mean(uni_bead_pcts):>10.1f}%" if uni_bead_pcts else f"{'N/A':>11s}"
            k_str = f"{np.mean(km_bead_pcts):>10.1f}%" if km_bead_pcts else f"{'N/A':>11s}"
            h_str = f"{np.mean(hi_bead_pcts):>10.1f}%" if hi_bead_pcts else f"{'N/A':>11s}"
            print(f"  {rname:<15s} {a_str} {u_str} {k_str} {h_str}")

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
