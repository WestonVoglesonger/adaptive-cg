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
    print("Analyzing results")
    print("  (not yet implemented)")
    return 0
