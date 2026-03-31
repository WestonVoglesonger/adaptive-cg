"""Exp 2: Multi-conformer validation across NMR ensembles.

Loads each model from multi-model PDB files (NMR ensembles), runs
uniform baselines + variable grid search on each conformer, and reports
whether variable CG improvement is consistent across conformers.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

# NMR ensemble PDB IDs known to contain multiple models.
DEFAULT_NMR_IDS = [
    "1D3Z",  # ubiquitin NMR, 10 models
    "2JOF",  # WW domain NMR peptide
    "1L2Y",  # trp-cage, 38 models
    "1GB1",  # protein G NMR variant
]


def setup_parser(parser):
    parser.add_argument(
        "--pdb-ids",
        nargs="+",
        default=DEFAULT_NMR_IDS,
        help=f"NMR ensemble PDB IDs to evaluate (default: {' '.join(DEFAULT_NMR_IDS)})",
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
        default=Path("results/conformer"),
        help="Output directory for results (default: results/conformer/)",
    )
    parser.add_argument(
        "--uniform-ratios",
        nargs="+",
        type=int,
        default=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        help="Uniform atom-to-bead ratios to test (default: 3-12)",
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
        "--fetch",
        action="store_true",
        help="Auto-fetch missing PDB files before running",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-conformer details",
    )


def _split_nmr_models(pdb_path: Path, work_dir: Path) -> list[Path]:
    """Split a multi-model PDB into one file per MODEL using BioPython.

    Returns a list of per-model PDB file paths.
    """
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Structure import Structure

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("nmr", str(pdb_path))
    models = list(structure.get_models())

    if len(models) <= 1:
        return [pdb_path]

    work_dir.mkdir(parents=True, exist_ok=True)
    stem = pdb_path.stem.upper()
    model_paths = []
    io = PDBIO()

    for model in models:
        model_id = model.get_id()
        out_path = work_dir / f"{stem}_model{model_id}.pdb"
        s = Structure(f"{stem}_m{model_id}")
        s.add(model.copy())
        io.set_structure(s)
        io.save(str(out_path))
        model_paths.append(out_path)

    return model_paths


def _evaluate_single_conformer(
    pdb_path: Path,
    uniform_ratios: list[int],
    grid_ratio_range: tuple[int, int],
) -> dict | None:
    """Run uniform + variable evaluation on one conformer file.

    Returns a result dict or None on failure.
    """
    from adaptive_cg.core.molecule import load_molecule
    from adaptive_cg.core.mapping import (
        eval_uniform_baselines,
        grid_search_variable,
    )

    try:
        mol = load_molecule(pdb_path)
    except Exception as e:
        print(f"    LOAD FAILED: {e}")
        return None

    # -- Uniform baselines --
    uniform_results = eval_uniform_baselines(
        mol.positions, mol.masses, uniform_ratios,
    )
    best_uniform = min(uniform_results, key=lambda r: r["rmse"])

    # -- Variable mapping --
    target_beads = best_uniform["n_beads"]
    tol = max(3, target_beads // 10)
    variable_result = grid_search_variable(
        mol.positions,
        mol.masses,
        mol.region_labels,
        mol.region_names,
        ratio_range=tuple(grid_ratio_range),
        target_beads=target_beads,
        tolerance=tol,
    )

    has_variable = variable_result["best_rmse"] < float("inf")

    result = {
        "n_atoms": mol.n_atoms,
        "mol_type": mol.mol_type,
        "region_names": mol.region_names,
        "region_counts": mol.region_counts,
        "best_uniform_ratio": best_uniform["ratio"],
        "best_uniform_beads": best_uniform["n_beads"],
        "best_uniform_rmse": best_uniform["rmse"],
    }

    if has_variable:
        result["variable_rmse"] = variable_result["best_rmse"]
        result["variable_ratios"] = variable_result["best_ratios"]
        result["variable_beads"] = variable_result["n_beads"]
        improvement = best_uniform["rmse"] - variable_result["best_rmse"]
        pct = (improvement / best_uniform["rmse"]) * 100 if best_uniform["rmse"] > 0 else 0.0
        result["improvement_nm"] = improvement
        result["improvement_pct"] = pct
    else:
        result["variable_rmse"] = None
        result["improvement_nm"] = None
        result["improvement_pct"] = None

    return result


def execute(args):
    import gc

    pdb_ids = [pid.upper() for pid in args.pdb_ids]
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional auto-fetch
    if args.fetch:
        from Bio.PDB import PDBList
        fetcher = PDBList(verbose=False)
        data_dir.mkdir(parents=True, exist_ok=True)
        for pdb_id in pdb_ids:
            out = data_dir / f"{pdb_id}.pdb"
            if out.exists():
                continue
            print(f"  Fetching {pdb_id}...")
            path = fetcher.retrieve_pdb_file(
                pdb_id, pdir=str(data_dir), file_format="pdb",
            )
            if path and Path(path).exists():
                Path(path).rename(out)

    print(f"Exp 2: Multi-conformer validation")
    print(f"  PDB IDs: {pdb_ids}")
    print(f"  Uniform ratios: {args.uniform_ratios}")
    print(f"  Grid ratio range: {args.grid_ratio_range[0]}-{args.grid_ratio_range[1]}")
    print()

    all_mol_results = []

    for pdb_id in pdb_ids:
        pdb_file = data_dir / f"{pdb_id}.pdb"
        if not pdb_file.exists():
            pdb_file = data_dir / f"{pdb_id.lower()}.pdb"
        if not pdb_file.exists():
            print(f"[{pdb_id}] PDB file not found, skipping")
            continue

        print(f"[{pdb_id}] Splitting NMR models...")
        work_dir = output_dir / pdb_id / "models"
        model_paths = _split_nmr_models(pdb_file, work_dir)
        n_models = len(model_paths)

        if n_models <= 1:
            print(f"  Only 1 model found (not an NMR ensemble), skipping")
            continue

        print(f"  Found {n_models} conformers")

        conformer_results = []
        for j, model_path in enumerate(model_paths):
            print(f"  Conformer {j}/{n_models-1} ... ", end="", flush=True)
            result = _evaluate_single_conformer(
                model_path,
                args.uniform_ratios,
                tuple(args.grid_ratio_range),
            )
            if result is None:
                print("FAILED")
                continue

            conformer_results.append(result)

            imp_str = (
                f"imp={result['improvement_pct']:.1f}%"
                if result["improvement_pct"] is not None
                else "var=N/A"
            )
            print(
                f"uni={result['best_uniform_rmse']:.4f}  "
                f"{imp_str}"
            )

            if args.verbose and result["variable_rmse"] is not None:
                print(
                    f"    variable_rmse={result['variable_rmse']:.6f}  "
                    f"ratios={result['variable_ratios']}"
                )

            gc.collect()

        if not conformer_results:
            print(f"  No conformers evaluated successfully")
            continue

        # ----------------------------------------------------------
        # Aggregate statistics across conformers
        # ----------------------------------------------------------
        improvements = [
            r["improvement_pct"]
            for r in conformer_results
            if r["improvement_pct"] is not None
        ]

        uniform_rmses = [r["best_uniform_rmse"] for r in conformer_results]
        variable_rmses = [
            r["variable_rmse"]
            for r in conformer_results
            if r["variable_rmse"] is not None
        ]

        summary = {
            "pdb_id": pdb_id,
            "n_conformers": len(conformer_results),
            "n_with_variable": len(improvements),
            "uniform_rmse_mean": float(np.mean(uniform_rmses)),
            "uniform_rmse_std": float(np.std(uniform_rmses)),
        }

        if improvements:
            imp_arr = np.array(improvements)
            summary["improvement_pct_mean"] = float(np.mean(imp_arr))
            summary["improvement_pct_std"] = float(np.std(imp_arr))
            summary["improvement_pct_min"] = float(np.min(imp_arr))
            summary["improvement_pct_max"] = float(np.max(imp_arr))
            summary["all_positive"] = bool(np.all(imp_arr > 0))
            summary["consistent"] = bool(np.std(imp_arr) < np.mean(imp_arr) * 0.5)

            var_arr = np.array(variable_rmses)
            summary["variable_rmse_mean"] = float(np.mean(var_arr))
            summary["variable_rmse_std"] = float(np.std(var_arr))

        print(f"\n  --- {pdb_id} Summary ({len(conformer_results)} conformers) ---")
        print(f"  Uniform RMSE:  {summary['uniform_rmse_mean']:.6f} +/- {summary['uniform_rmse_std']:.6f}")

        if improvements:
            print(f"  Variable RMSE: {summary['variable_rmse_mean']:.6f} +/- {summary['variable_rmse_std']:.6f}")
            print(
                f"  Improvement:   {summary['improvement_pct_mean']:.1f}% "
                f"+/- {summary['improvement_pct_std']:.1f}%  "
                f"(range: {summary['improvement_pct_min']:.1f}% to {summary['improvement_pct_max']:.1f}%)"
            )
            print(f"  All positive:  {summary['all_positive']}")
            print(f"  Consistent:    {summary['consistent']}  (std < 0.5*mean)")
        else:
            print(f"  No variable mappings found for any conformer")

        print()

        # ----------------------------------------------------------
        # Save per-molecule JSON
        # ----------------------------------------------------------
        mol_output = {
            "pdb_id": pdb_id,
            "summary": summary,
            "conformers": conformer_results,
        }
        mol_dir = output_dir / pdb_id
        mol_dir.mkdir(parents=True, exist_ok=True)
        out_file = mol_dir / "conformer_result.json"
        with open(out_file, "w") as f:
            json.dump(mol_output, f, indent=2, default=str)
        print(f"  Saved: {out_file}")

        all_mol_results.append(mol_output)
        gc.collect()

    # ------------------------------------------------------------------
    # Aggregate summary across all molecules
    # ------------------------------------------------------------------
    if not all_mol_results:
        print("\nNo molecules evaluated.")
        return 1

    print(f"\n{'='*60}")
    print(f"Exp 2 Aggregate: {len(all_mol_results)} NMR ensembles")
    print(f"{'='*60}")

    all_improvements = []
    for mol in all_mol_results:
        s = mol["summary"]
        n_conf = s["n_conformers"]
        imp_str = (
            f"{s['improvement_pct_mean']:.1f}% +/- {s['improvement_pct_std']:.1f}%"
            if "improvement_pct_mean" in s
            else "N/A"
        )
        consistent = s.get("consistent", "N/A")
        print(f"  {s['pdb_id']:6s}  {n_conf:2d} conformers  improvement: {imp_str}  consistent: {consistent}")

        if "improvement_pct_mean" in s:
            all_improvements.append(s["improvement_pct_mean"])

    if all_improvements:
        arr = np.array(all_improvements)
        print(f"\n  Mean improvement across ensembles: {arr.mean():.1f}%")
        print(f"  All ensembles show improvement: {bool(np.all(arr > 0))}")

    # Save aggregate JSON
    agg_file = output_dir / "conformer_aggregate.json"
    with open(agg_file, "w") as f:
        json.dump(all_mol_results, f, indent=2, default=str)
    print(f"\n  Aggregate saved: {agg_file}")

    return 0
