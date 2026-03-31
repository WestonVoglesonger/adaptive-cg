"""Compare all CG mapping optimization strategies head-to-head."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path


def setup_parser(parser):
    parser.add_argument(
        "pdb_ids",
        nargs="*",
        help=(
            "PDB IDs to compare (e.g. 1UBQ 1CRN). "
            "If omitted, reads from results/sweep_summary.csv"
        ),
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
        default=Path("results/comparison"),
        help="Output directory (default: results/comparison/)",
    )
    parser.add_argument(
        "--n-beads",
        type=int,
        default=None,
        help="Fixed bead count (default: use best uniform bead count per molecule)",
    )
    parser.add_argument(
        "--skip-annealing",
        action="store_true",
        help="Skip simulated annealing (slow)",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=1000,
        help="Skip molecules with more than N heavy atoms (default: 1000)",
    )


def _load_pdb_ids_from_sweep(results_dir: Path) -> list[str]:
    """Read PDB IDs from sweep_summary.csv."""
    csv_path = results_dir / "sweep_summary.csv"
    if not csv_path.exists():
        return []
    ids = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row.get("pdb_id", "").strip()
            if pdb_id:
                ids.append(pdb_id)
    return ids


def execute(args):
    import gc

    import numpy as np
    from scipy.spatial.distance import cdist

    from adaptive_cg.core.molecule import load_molecule
    from adaptive_cg.core.mapping import (
        eval_mapping,
        eval_uniform_baselines,
        generate_uniform_mapping,
        grid_search_variable,
    )

    # Guarded imports for optional dependencies
    sklearn_available = True
    try:
        from adaptive_cg.core.strategies import (
            kmeans_mapping,
            spectral_mapping,
            hierarchical_mapping,
        )
    except ImportError:
        sklearn_available = False
        kmeans_mapping = spectral_mapping = hierarchical_mapping = None

    annealing_available = True
    try:
        from adaptive_cg.core.strategies import annealing_mapping
    except ImportError:
        annealing_available = False
        annealing_mapping = None

    torch_available = True
    try:
        from adaptive_cg.core.optimizer import DifferentiableCGOptimizer
    except ImportError:
        torch_available = False
        DifferentiableCGOptimizer = None

    # Resolve molecule list
    pdb_ids = args.pdb_ids
    if not pdb_ids:
        pdb_ids = _load_pdb_ids_from_sweep(Path("results"))
    if not pdb_ids:
        print("No PDB IDs provided and no sweep_summary.csv found.")
        print("Run 'acg sweep' first or pass PDB IDs as arguments.")
        return 1

    pdb_ids = [pid.upper() for pid in pdb_ids]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build strategy roster
    strategy_names = ["uniform", "grid_search"]
    if sklearn_available:
        strategy_names.extend(["kmeans", "spectral", "hierarchical"])
    else:
        print("Warning: scikit-learn not installed, skipping kmeans/spectral/hierarchical")
    if annealing_available and not args.skip_annealing:
        strategy_names.append("annealing")
    elif args.skip_annealing:
        print("Skipping simulated annealing (--skip-annealing)")
    if torch_available:
        strategy_names.append("pytorch")
    else:
        print("Warning: PyTorch not installed, skipping pytorch optimizer")

    print(f"Comparing {len(strategy_names)} strategies across {len(pdb_ids)} molecules")
    print(f"  Strategies: {', '.join(strategy_names)}")
    print(f"  Max atoms: {args.max_atoms}")
    print()

    # Prepare CSV for incremental writes
    csv_file = output_dir / "comparison_summary.csv"
    csv_fieldnames = ["pdb_id", "n_atoms", "n_beads", "strategy", "rmse", "time_s"]
    csv_fh = open(csv_file, "w", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=csv_fieldnames)
    csv_writer.writeheader()
    csv_fh.flush()

    all_results = []  # list of dicts for JSON output
    n_success, n_skip = 0, 0

    for mol_idx, pdb_id in enumerate(pdb_ids, 1):
        print(f"[{mol_idx}/{len(pdb_ids)}] {pdb_id} ... ", end="", flush=True)

        # Locate PDB file
        pdb_file = args.data_dir / f"{pdb_id}.pdb"
        if not pdb_file.exists():
            pdb_file = args.data_dir / f"{pdb_id.lower()}.pdb"
        if not pdb_file.exists():
            print("NOT FOUND")
            n_skip += 1
            continue

        try:
            mol = load_molecule(pdb_file)
        except Exception as e:
            print(f"LOAD FAILED: {e}")
            n_skip += 1
            continue

        if mol.n_atoms > args.max_atoms:
            print(f"SKIP ({mol.n_atoms} atoms > {args.max_atoms})")
            n_skip += 1
            continue

        # Determine target bead count
        if args.n_beads is not None:
            n_beads = args.n_beads
        else:
            # Find best uniform bead count
            uniform_results = eval_uniform_baselines(
                mol.positions, mol.masses, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            )
            best_uniform = min(uniform_results, key=lambda r: r["rmse"])
            n_beads = best_uniform["n_beads"]

        if n_beads < 2:
            print(f"SKIP (n_beads={n_beads} < 2)")
            n_skip += 1
            continue

        print(f"{mol.n_atoms} atoms, {n_beads} beads")

        # Pre-compute all-atom distance matrix once
        aa_dmat = cdist(mol.positions, mol.positions) if mol.n_atoms <= 1000 else None

        mol_results = {}

        # --- Uniform ---
        if "uniform" in strategy_names:
            t0 = time.time()
            ratio = max(1, mol.n_atoms // n_beads)
            mapping = generate_uniform_mapping(mol.n_atoms, ratio)
            result = eval_mapping(mapping, mol.positions, mol.masses, aa_dmat=aa_dmat)
            elapsed = time.time() - t0
            mol_results["uniform"] = {"rmse": result["rmse"], "time_s": elapsed}

        # --- Grid search ---
        if "grid_search" in strategy_names:
            t0 = time.time()
            tol = max(3, n_beads // 10)
            gs_result = grid_search_variable(
                mol.positions, mol.masses,
                mol.region_labels, mol.region_names,
                target_beads=n_beads, tolerance=tol,
            )
            elapsed = time.time() - t0
            rmse = gs_result["best_rmse"] if gs_result["best_rmse"] < float("inf") else None
            mol_results["grid_search"] = {"rmse": rmse, "time_s": elapsed}

        # --- K-means ---
        if "kmeans" in strategy_names and sklearn_available:
            t0 = time.time()
            try:
                mapping = kmeans_mapping(mol.positions, mol.masses, n_beads)
                result = eval_mapping(mapping, mol.positions, mol.masses, aa_dmat=aa_dmat)
                elapsed = time.time() - t0
                mol_results["kmeans"] = {"rmse": result["rmse"], "time_s": elapsed}
            except Exception as e:
                elapsed = time.time() - t0
                print(f"    kmeans failed: {e}")
                mol_results["kmeans"] = {"rmse": None, "time_s": elapsed}

        # --- Spectral ---
        if "spectral" in strategy_names and sklearn_available:
            t0 = time.time()
            try:
                mapping = spectral_mapping(mol.positions, mol.masses, n_beads)
                result = eval_mapping(mapping, mol.positions, mol.masses, aa_dmat=aa_dmat)
                elapsed = time.time() - t0
                mol_results["spectral"] = {"rmse": result["rmse"], "time_s": elapsed}
            except Exception as e:
                elapsed = time.time() - t0
                print(f"    spectral failed: {e}")
                mol_results["spectral"] = {"rmse": None, "time_s": elapsed}

        # --- Hierarchical ---
        if "hierarchical" in strategy_names and sklearn_available:
            t0 = time.time()
            try:
                mapping = hierarchical_mapping(mol.positions, mol.masses, n_beads)
                result = eval_mapping(mapping, mol.positions, mol.masses, aa_dmat=aa_dmat)
                elapsed = time.time() - t0
                mol_results["hierarchical"] = {"rmse": result["rmse"], "time_s": elapsed}
            except Exception as e:
                elapsed = time.time() - t0
                print(f"    hierarchical failed: {e}")
                mol_results["hierarchical"] = {"rmse": None, "time_s": elapsed}

        # --- Simulated annealing ---
        if "annealing" in strategy_names and annealing_available:
            t0 = time.time()
            try:
                mapping = annealing_mapping(mol.positions, mol.masses, n_beads)
                result = eval_mapping(mapping, mol.positions, mol.masses, aa_dmat=aa_dmat)
                elapsed = time.time() - t0
                mol_results["annealing"] = {"rmse": result["rmse"], "time_s": elapsed}
            except Exception as e:
                elapsed = time.time() - t0
                print(f"    annealing failed: {e}")
                mol_results["annealing"] = {"rmse": None, "time_s": elapsed}

        # --- PyTorch optimizer ---
        if "pytorch" in strategy_names and torch_available and mol.n_atoms <= 200:
            t0 = time.time()
            try:
                opt = DifferentiableCGOptimizer(
                    positions=mol.positions,
                    masses=mol.masses,
                    n_beads=n_beads,
                    epochs=1000,
                )
                opt_result = opt.optimize()
                elapsed = time.time() - t0
                mol_results["pytorch"] = {"rmse": opt_result["rmse"], "time_s": elapsed}
            except Exception as e:
                elapsed = time.time() - t0
                print(f"    pytorch failed: {e}")
                mol_results["pytorch"] = {"rmse": None, "time_s": elapsed}
        elif "pytorch" in strategy_names and mol.n_atoms > 200:
            mol_results["pytorch"] = {"rmse": None, "time_s": 0.0}

        # Print per-molecule comparison table
        print(f"  {'Strategy':<16} {'RMSE':>12} {'Time (s)':>10}")
        print(f"  {'-'*16} {'-'*12} {'-'*10}")
        for strat in strategy_names:
            if strat in mol_results:
                r = mol_results[strat]
                rmse_str = f"{r['rmse']:.6f}" if r["rmse"] is not None else "N/A"
                print(f"  {strat:<16} {rmse_str:>12} {r['time_s']:>10.2f}")

        # Find best strategy for this molecule
        valid = {k: v for k, v in mol_results.items() if v["rmse"] is not None}
        if valid:
            best_strat = min(valid, key=lambda k: valid[k]["rmse"])
            best_rmse = valid[best_strat]["rmse"]
            print(f"  -> Best: {best_strat} (RMSE={best_rmse:.6f})")
        print()

        # Write CSV rows incrementally
        for strat, res in mol_results.items():
            row = {
                "pdb_id": pdb_id,
                "n_atoms": mol.n_atoms,
                "n_beads": n_beads,
                "strategy": strat,
                "rmse": res["rmse"],
                "time_s": round(res["time_s"], 3),
            }
            csv_writer.writerow(row)
        csv_fh.flush()

        # Collect for JSON
        all_results.append({
            "pdb_id": pdb_id,
            "n_atoms": mol.n_atoms,
            "n_beads": n_beads,
            "strategies": mol_results,
        })

        n_success += 1

        # Free memory
        del mol, mol_results, aa_dmat
        gc.collect()

    csv_fh.close()

    # ------------------------------------------------------------------
    # Aggregate tables
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Comparison complete: {n_success} molecules evaluated, {n_skip} skipped")

    if not all_results:
        return 0

    # Aggregate: mean RMSE by strategy
    strategy_rmses: dict[str, list[float]] = {s: [] for s in strategy_names}
    strategy_wins: dict[str, int] = {s: 0 for s in strategy_names}

    for mol_data in all_results:
        valid = {}
        for strat, res in mol_data["strategies"].items():
            if res["rmse"] is not None:
                strategy_rmses[strat].append(res["rmse"])
                valid[strat] = res["rmse"]
        if valid:
            best = min(valid, key=lambda k: valid[k])
            strategy_wins[best] = strategy_wins.get(best, 0) + 1

    import numpy as np

    print(f"\n{'Strategy':<16} {'Mean RMSE':>12} {'Median RMSE':>14} {'N':>5} {'Wins':>6}")
    print(f"{'-'*16} {'-'*12} {'-'*14} {'-'*5} {'-'*6}")
    for strat in strategy_names:
        vals = strategy_rmses[strat]
        if vals:
            arr = np.array(vals)
            print(
                f"{strat:<16} {arr.mean():>12.6f} {np.median(arr):>14.6f} "
                f"{len(vals):>5d} {strategy_wins.get(strat, 0):>6d}"
            )
        else:
            print(f"{strat:<16} {'N/A':>12} {'N/A':>14} {'0':>5} {'0':>6}")

    # Print "best strategy wins" summary
    print(f"\nBest strategy wins:")
    for strat in sorted(strategy_wins, key=lambda k: strategy_wins[k], reverse=True):
        if strategy_wins[strat] > 0:
            print(f"  {strat}: {strategy_wins[strat]}")

    # Save JSON
    json_file = output_dir / "comparison_summary.json"
    # Convert for JSON serialization (strip numpy types)
    serializable = []
    for entry in all_results:
        s_entry = {
            "pdb_id": entry["pdb_id"],
            "n_atoms": int(entry["n_atoms"]),
            "n_beads": int(entry["n_beads"]),
            "strategies": {},
        }
        for strat, res in entry["strategies"].items():
            s_entry["strategies"][strat] = {
                "rmse": float(res["rmse"]) if res["rmse"] is not None else None,
                "time_s": float(res["time_s"]),
            }
        serializable.append(s_entry)

    with open(json_file, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  CSV:  {csv_file}")
    print(f"  JSON: {json_file}")

    return 0
