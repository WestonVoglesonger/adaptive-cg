"""Aggregate results, compute statistics, generate plots."""
from pathlib import Path


def setup_parser(parser):
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory (default: results/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis"),
        help="Output directory for plots and tables (default: analysis/)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Significance level for statistical tests (default: 0.01)",
    )


def execute(args):
    import csv
    import json

    import numpy as np
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = args.results_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = results_dir / "sweep_summary.csv"
    if not csv_file.exists():
        print(f"No sweep_summary.csv found in {results_dir}/")
        print("Run 'acg sweep' first.")
        return 1

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    rows = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for key in ("n_atoms", "best_uniform_beads", "variable_beads", "optimize_beads"):
                if r.get(key):
                    r[key] = int(float(r[key]))
            for key in ("best_uniform_rmse", "best_uniform_mae",
                        "variable_rmse", "improvement_nm", "improvement_pct",
                        "optimize_rmse"):
                if r.get(key):
                    try:
                        r[key] = float(r[key])
                    except (ValueError, TypeError):
                        r[key] = None
            rows.append(r)

    print(f"Loaded {len(rows)} molecules from {csv_file}")
    print()

    # ------------------------------------------------------------------
    # Split by molecule type
    # ------------------------------------------------------------------
    proteins = [r for r in rows if r["mol_type"] == "protein"]
    nucleic = [r for r in rows if r["mol_type"] == "nucleic_acid"]
    other = [r for r in rows if r["mol_type"] not in ("protein", "nucleic_acid")]

    # Rows with valid improvement data
    has_var = [r for r in rows if r["improvement_pct"] is not None]
    has_opt = [r for r in rows if r.get("optimize_rmse") is not None]

    # ------------------------------------------------------------------
    # 1. Summary table
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 1 RESULTS: Variable vs Uniform CG Mapping")
    print("=" * 70)
    print()

    def print_category(name, subset):
        vals = [r for r in subset if r["improvement_pct"] is not None]
        if not vals:
            print(f"  {name}: no data")
            return
        pcts = np.array([r["improvement_pct"] for r in vals])
        wins = (pcts > 0).sum()
        print(f"  {name} ({len(vals)} molecules):")
        print(f"    Mean improvement:   {pcts.mean():+.1f}%")
        print(f"    Median improvement: {np.median(pcts):+.1f}%")
        print(f"    Min / Max:          {pcts.min():+.1f}% / {pcts.max():+.1f}%")
        print(f"    Wins / Total:       {wins}/{len(vals)}")
        print()

    print_category("All molecules", rows)
    print_category("Proteins", proteins)
    print_category("Nucleic acids", nucleic)
    if other:
        print_category("Other", other)

    # ------------------------------------------------------------------
    # 2. Statistical tests
    # ------------------------------------------------------------------
    print("-" * 70)
    print("STATISTICAL TESTS")
    print("-" * 70)
    print()

    if len(has_var) >= 5:
        uni_rmse = np.array([r["best_uniform_rmse"] for r in has_var])
        var_rmse = np.array([r["variable_rmse"] for r in has_var])

        # Paired Wilcoxon signed-rank test (one-sided: variable < uniform)
        diffs = uni_rmse - var_rmse
        stat_w, p_wilcoxon_two = stats.wilcoxon(diffs)
        # One-sided: we expect variable to be better (diffs > 0)
        positive = (diffs > 0).sum()
        p_wilcoxon = p_wilcoxon_two / 2 if positive > len(diffs) / 2 else 1 - p_wilcoxon_two / 2

        # Paired t-test
        t_stat, p_ttest_two = stats.ttest_rel(uni_rmse, var_rmse)
        p_ttest = p_ttest_two / 2 if t_stat > 0 else 1 - p_ttest_two / 2

        sig_w = "YES" if p_wilcoxon < args.alpha else "NO"
        sig_t = "YES" if p_ttest < args.alpha else "NO"

        print(f"  All molecules with variable mapping (n={len(has_var)}):")
        print(f"    Wilcoxon signed-rank (one-sided): p = {p_wilcoxon:.6f}  Significant at α={args.alpha}? {sig_w}")
        print(f"    Paired t-test (one-sided):        p = {p_ttest:.6f}  Significant at α={args.alpha}? {sig_t}")
        print(f"    Mean RMSE reduction: {diffs.mean():.6f} nm")
        print(f"    Positive diffs (var wins): {positive}/{len(diffs)}")
        print()

        # By category
        for name, subset in [("Proteins", proteins), ("Nucleic acids", nucleic)]:
            cat_has = [r for r in subset if r["improvement_pct"] is not None]
            if len(cat_has) < 5:
                continue
            u = np.array([r["best_uniform_rmse"] for r in cat_has])
            v = np.array([r["variable_rmse"] for r in cat_has])
            d = u - v
            pos = (d > 0).sum()
            try:
                _, pw2 = stats.wilcoxon(d)
                pw = pw2 / 2 if pos > len(d) / 2 else 1 - pw2 / 2
            except ValueError:
                pw = 1.0
            sig = "YES" if pw < args.alpha else "NO"
            print(f"  {name} (n={len(cat_has)}):")
            print(f"    Wilcoxon (one-sided): p = {pw:.6f}  Significant? {sig}")
            print(f"    Mean RMSE reduction: {d.mean():.6f} nm ({(d/u*100).mean():.1f}%)")
            print(f"    Wins: {pos}/{len(d)}")
            print()

    # ------------------------------------------------------------------
    # 3. PyTorch optimizer comparison
    # ------------------------------------------------------------------
    if has_opt:
        print("-" * 70)
        print("PYTORCH OPTIMIZER RESULTS")
        print("-" * 70)
        print()
        for r in sorted(has_opt, key=lambda x: x["pdb_id"]):
            uni = r["best_uniform_rmse"]
            opt = r["optimize_rmse"]
            var = r.get("variable_rmse")
            opt_imp = (uni - opt) / uni * 100 if uni > 0 else 0
            best_label = "opt" if (var is None or opt < var) else "var"
            var_str = f"var={var:.4f}" if var else "var=N/A"
            print(f"  {r['pdb_id']:6s}: uni={uni:.4f}  {var_str}  opt={opt:.4f}  "
                  f"opt_vs_uni={opt_imp:+.1f}%  best={best_label}")
        print()

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    print("-" * 70)
    print("GENERATING PLOTS")
    print("-" * 70)
    print()

    # --- Plot 1: Improvement by molecule ---
    if has_var:
        sorted_rows = sorted(has_var, key=lambda r: r["improvement_pct"])
        names = [r["pdb_id"] for r in sorted_rows]
        pcts = [r["improvement_pct"] for r in sorted_rows]
        types = [r["mol_type"] for r in sorted_rows]
        colors = ["#2196F3" if t == "nucleic_acid" else "#FF9800" for t in types]

        fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.35)))
        bars = ax.barh(range(len(names)), pcts, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Improvement over best uniform (%)")
        ax.set_title("Variable vs Uniform CG Mapping: Improvement by Molecule")
        ax.axvline(0, color="black", linewidth=0.5)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#FF9800", label="Protein"),
            Patch(facecolor="#2196F3", label="Nucleic acid"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        path = output_dir / "improvement_by_molecule.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # --- Plot 2: RMSE scatter (uniform vs variable) ---
    if has_var:
        fig, ax = plt.subplots(figsize=(8, 8))
        for r in has_var:
            c = "#2196F3" if r["mol_type"] == "nucleic_acid" else "#FF9800"
            ax.scatter(r["best_uniform_rmse"], r["variable_rmse"],
                       c=c, s=40, alpha=0.8, edgecolors="black", linewidth=0.5)
            ax.annotate(r["pdb_id"], (r["best_uniform_rmse"], r["variable_rmse"]),
                        fontsize=5, alpha=0.7, xytext=(2, 2), textcoords="offset points")

        lim_max = max(
            max(r["best_uniform_rmse"] for r in has_var),
            max(r["variable_rmse"] for r in has_var),
        ) * 1.1
        # Exclude extreme outliers from axis limits
        non_outlier = [r for r in has_var
                       if r["best_uniform_rmse"] < 0.1 and r["variable_rmse"] < 0.1]
        if non_outlier:
            lim_max = max(
                max(r["best_uniform_rmse"] for r in non_outlier),
                max(r["variable_rmse"] for r in non_outlier),
            ) * 1.15

        ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.3, label="y=x (no improvement)")
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_xlabel("Best Uniform RMSE (nm)")
        ax.set_ylabel("Best Variable RMSE (nm)")
        ax.set_title("Uniform vs Variable CG: Distance Matrix RMSE")
        ax.set_aspect("equal")
        legend_elements = [
            Patch(facecolor="#FF9800", label="Protein"),
            Patch(facecolor="#2196F3", label="Nucleic acid"),
        ]
        ax.legend(handles=legend_elements)

        plt.tight_layout()
        path = output_dir / "rmse_scatter.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # --- Plot 3: Improvement distribution ---
    if has_var:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        prot_pcts = [r["improvement_pct"] for r in proteins if r["improvement_pct"] is not None]
        nuc_pcts = [r["improvement_pct"] for r in nucleic if r["improvement_pct"] is not None]

        if prot_pcts:
            axes[0].hist(prot_pcts, bins=15, color="#FF9800", edgecolor="black", alpha=0.8)
            axes[0].axvline(np.mean(prot_pcts), color="red", linestyle="--",
                            label=f"Mean: {np.mean(prot_pcts):.1f}%")
            axes[0].axvline(0, color="black", linewidth=0.5)
            axes[0].set_xlabel("Improvement (%)")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Proteins")
            axes[0].legend()

        if nuc_pcts:
            axes[1].hist(nuc_pcts, bins=10, color="#2196F3", edgecolor="black", alpha=0.8)
            axes[1].axvline(np.mean(nuc_pcts), color="red", linestyle="--",
                            label=f"Mean: {np.mean(nuc_pcts):.1f}%")
            axes[1].axvline(0, color="black", linewidth=0.5)
            axes[1].set_xlabel("Improvement (%)")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Nucleic Acids")
            axes[1].legend()

        plt.suptitle("Distribution of Variable vs Uniform Improvement")
        plt.tight_layout()
        path = output_dir / "improvement_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # --- Plot 4: Improvement vs molecule size ---
    if has_var:
        fig, ax = plt.subplots(figsize=(10, 6))
        for r in has_var:
            c = "#2196F3" if r["mol_type"] == "nucleic_acid" else "#FF9800"
            ax.scatter(r["n_atoms"], r["improvement_pct"],
                       c=c, s=50, alpha=0.8, edgecolors="black", linewidth=0.5)
            ax.annotate(r["pdb_id"], (r["n_atoms"], r["improvement_pct"]),
                        fontsize=6, alpha=0.6, xytext=(3, 3), textcoords="offset points")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Number of heavy atoms")
        ax.set_ylabel("Improvement over uniform (%)")
        ax.set_title("Variable CG Improvement vs Molecule Size")
        legend_elements = [
            Patch(facecolor="#FF9800", label="Protein"),
            Patch(facecolor="#2196F3", label="Nucleic acid"),
        ]
        ax.legend(handles=legend_elements)

        plt.tight_layout()
        path = output_dir / "improvement_vs_size.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ------------------------------------------------------------------
    # 5. Save analysis JSON
    # ------------------------------------------------------------------
    analysis = {
        "n_molecules": len(rows),
        "n_proteins": len(proteins),
        "n_nucleic_acids": len(nucleic),
        "n_with_variable": len(has_var),
        "n_with_optimizer": len(has_opt),
    }

    if has_var:
        pcts = np.array([r["improvement_pct"] for r in has_var])
        analysis["overall"] = {
            "mean_improvement_pct": float(pcts.mean()),
            "median_improvement_pct": float(np.median(pcts)),
            "std_improvement_pct": float(pcts.std()),
            "min_improvement_pct": float(pcts.min()),
            "max_improvement_pct": float(pcts.max()),
            "n_positive": int((pcts > 0).sum()),
        }
        if len(has_var) >= 5:
            analysis["overall"]["wilcoxon_p"] = float(p_wilcoxon)
            analysis["overall"]["ttest_p"] = float(p_ttest)
            analysis["overall"]["significant_at_alpha"] = float(args.alpha)

    analysis_file = output_dir / "analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis JSON: {analysis_file}")

    print()
    print("=" * 70)
    print("Analysis complete.")
    print("=" * 70)

    return 0
