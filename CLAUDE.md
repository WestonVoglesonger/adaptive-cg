# Adaptive Resolution Coarse-Graining

## Project

CLI tool (`acg`) for adaptive resolution coarse-grained molecular dynamics. The mapping problem (which atoms group into which beads) is solved — k-means clustering produces near-optimal mappings in <1 second. Next phase: CG simulation engine.

Repo: github.com/WestonVoglesonger/adaptive-cg
Runs in GitHub Codespace (M1 Mac OOMs on large molecules).

## Architecture

```
src/adaptive_cg/
  __init__.py          # CLI entrypoint, argparse dispatch (table-driven)
  __main__.py          # python -m adaptive_cg
  core/
    molecule.py        # MoleculeData, load_molecule (MDAnalysis + mkdssp)
    mapping.py         # eval_mapping, uniform/variable generators, grid search
    optimizer.py       # PyTorch differentiable soft-assignment (optional)
    strategies.py      # k-means, spectral, hierarchical, simulated annealing
  commands/
    fetch.py           # Download PDB structures (42 molecules across 4 categories)
    list_molecules.py  # Show downloaded structures
    evaluate.py        # Single molecule: uniform + variable evaluation
    optimize.py        # PyTorch optimization for one molecule
    sweep.py           # Phase 1: all molecules, incremental CSV output
    analyze.py         # Statistics (Wilcoxon, t-test), plots
    conformer.py       # Exp 2: NMR multi-conformer validation
    region_breakdown.py # Exp 3: per-region RMSE (helix/sheet/loop, base/sugar/phosphate)
    compare_optimizers.py # Head-to-head strategy comparison
    pareto.py          # Phase 2: accuracy vs compute cost (stub)
```

## Setup

```bash
pip install -e ".[optimize]"        # Core + PyTorch
pip install scikit-learn            # For k-means/spectral/hierarchical
conda install -c conda-forge dssp   # mkdssp binary for DSSP (required)
```

## Key conventions

- Positions are always in **nm** (PDB Angstroms / 10)
- A "mapping" is `list[list[int]]` — each inner list is atom indices for one bead
- `eval_mapping` computes RMSE between CG distance matrix and projected AA distance matrix
- Pre-compute `aa_dmat` via cdist and pass it to avoid recomputation in loops
- Sweep/compare commands write CSV incrementally (survives crashes)
- `gc.collect()` between molecules to manage memory
- mkdssp must not silently fall back — raise errors explicitly

## Key findings (completed)

- K-means/hierarchical clustering: ~74% RMSE improvement over uniform, 3x better than DSSP grid search
- The win is from **bead boundary placement**, not resolution redistribution
- Grid search only wins on nucleic acid phosphate backbone
- Validated across 39 molecules (p<0.001), 136 NMR conformers, per-region analysis
- PyTorch differentiable optimizer underperforms k-means (regularization fights reconstruction)

## What's next

- Pareto curves or skip if shape is predictable
- CG force field parameterization
- Simulation engine (integrator, thermostat)
- Entropy/thermodynamics layer (kept separate from static geometry)
