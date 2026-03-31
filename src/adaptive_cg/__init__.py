"""
Adaptive Resolution Coarse-Graining

Validates that variable atom-to-bead ratios in molecular coarse-graining
preserve structural properties better than fixed uniform ratios.
"""
import argparse
import sys

__version__ = "0.1.0"


def main():
    from adaptive_cg.commands import (
        fetch, evaluate, optimize, sweep, analyze, pareto, list_molecules,
        conformer, region_breakdown, compare_optimizers,
    )

    parser = argparse.ArgumentParser(
        prog="acg",
        description="Adaptive Resolution Coarse-Graining: validation & analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  acg fetch --category proteins           # Download protein structures from PDB
  acg list                                # Show downloaded molecules
  acg evaluate 1UBQ                       # Evaluate single molecule (uniform + variable)
  acg optimize 1UBQ --beads 100           # PyTorch differentiable optimization
  acg sweep                               # Run Phase 1 across all molecules
  acg analyze                             # Aggregate results, statistics, plots
  acg pareto                              # Phase 2 Pareto frontier curves
  acg conformer                            # Exp 2: multi-conformer NMR validation
  acg region-breakdown                    # Exp 3: per-region RMSE breakdown
  acg compare                             # Benchmark all strategies head-to-head
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    commands = {
        "fetch":    (fetch,           "Download molecular structures from PDB/Zenodo"),
        "list":     (list_molecules,  "List downloaded molecules"),
        "evaluate": (evaluate,        "Evaluate CG mappings for a single molecule"),
        "optimize": (optimize,        "PyTorch differentiable CG optimization"),
        "sweep":    (sweep,           "Run evaluation across all molecules (Phase 1)"),
        "analyze":  (analyze,         "Aggregate results, statistics, and plots"),
        "pareto":   (pareto,          "Generate Pareto frontier curves (Phase 2)"),
        "conformer": (conformer, "Multi-conformer NMR ensemble validation (Exp 2)"),
        "region-breakdown": (region_breakdown, "Per-region RMSE breakdown (Exp 3)"),
        "compare": (compare_optimizers, "Compare optimization strategies head-to-head"),
    }

    for name, (module, help_text) in commands.items():
        sub = subparsers.add_parser(name, help=help_text)
        module.setup_parser(sub)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        module = commands[args.command][0]
        sys.exit(module.execute(args))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
