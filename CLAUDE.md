# Adaptive Resolution Coarse-Graining

## Project

CLI tool (`acg`) for adaptive resolution coarse-grained molecular dynamics. Two phases completed:

1. **Mapping problem solved** — k-means clustering produces near-optimal CG mappings in <1 second
2. **CG simulation engine working** — full pipeline from PDB → force field → CG MD trajectory

Next phase: adaptive resolution controller (dynamic remapping during simulation).

Repo: github.com/WestonVoglesonger/adaptive-cg
AA simulations run in Google Colab (free T4 GPU). CG simulations run locally (M1 Mac).

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
    simulation.py      # OpenMM AA MD simulation for reference trajectories
    extract.py         # Map AA trajectory → CG distributions by bead class
    parameterize.py    # Boltzmann inversion → transferable CG force field
    engine.py          # CG MD engine: forces, Verlet integrator, Langevin thermostat
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
    simulate.py        # Run AA MD via OpenMM
    extract.py         # Extract CG distributions from AA trajectory
    parameterize.py    # Derive transferable CG force field
    cg_simulate.py     # Run CG MD simulation
  notebooks/
    aa_simulate_colab.ipynb  # Batch AA simulations on Colab GPU
```

## Setup

```bash
pip install -e ".[optimize,simulate]"  # Core + PyTorch + OpenMM
pip install scikit-learn               # For k-means/spectral/hierarchical
conda install -c conda-forge dssp      # mkdssp binary for DSSP (validation only)
```

## Full pipeline

```bash
# 1. Fetch molecules
acg fetch --category proteins

# 2. Run AA MD (Colab recommended for batch, or locally for single)
acg simulate 1UBQ                    # ~15 min locally, ~4 min Colab GPU

# 3. Extract CG distributions from AA trajectory
acg extract 1UBQ                     # Maps trajectory → bead class distributions

# 4. Derive transferable CG force field (pool across molecules)
acg parameterize                     # Boltzmann inversion → cg_forcefield.json

# 5. Run CG MD simulation
acg cg-simulate 1UBQ --steps 100000  # ~2 min for 1 ns, 150 beads
```

## Key conventions

- Positions are always in **nm** (PDB Angstroms / 10)
- A "mapping" is `list[list[int]]` — each inner list is atom indices for one bead
- `eval_mapping` computes RMSE between CG distance matrix and projected AA distance matrix
- Pre-compute `aa_dmat` via cdist and pass it to avoid recomputation in loops
- Sweep/compare commands write CSV incrementally (survives crashes)
- `gc.collect()` between molecules to manage memory
- CG force field is transferable: parameterized by bead chemical class, not per-molecule
- Bead classes: `{mol_type}_{polarity}_{size}` (e.g., protein_hydrophobic_M)
- LJ sigma derived from bead size (n_atoms^(1/3)), not from extracted RDF
- Non-bonded exclusions: 1-2, 1-3, and 1-4 neighbors excluded from LJ
- Energy minimization (steepest descent) runs before CG MD to remove bad contacts

## CG force field

Trained on 6 molecules (1CRN, 1UBQ, 1LYZ, 1BNA, 1IGD, 1PGA — 5 proteins + 1 DNA):
- 59 bond types (harmonic: k, r0)
- 348 angle types (harmonic: k, theta0)
- 52 non-bonded types (LJ: sigma from bead size, epsilon from Boltzmann inversion)
- Stored in `data/forcefield/cg_forcefield.json`

## Key findings (completed)

- K-means/hierarchical clustering: ~74% RMSE improvement over uniform, 3x better than DSSP grid search
- The win is from **bead boundary placement**, not resolution redistribution
- Grid search only wins on nucleic acid phosphate backbone
- Validated across 39 molecules (p<0.001), 136 NMR conformers, per-region analysis
- PyTorch differentiable optimizer underperforms k-means (regularization fights reconstruction)
- CG engine equilibrates to target temperature (~300-370 K) within ~50k steps

## What's next

- **Adaptive resolution controller** — the core contribution:
  - Activity monitor (RMSF, energy per region)
  - Decision function: when/where to add/remove beads
  - Remapping strategies: fixed, periodic with interpolation, continuous (AdResS-style)
  - Force field transition handling (avoid energy discontinuities)
- Entropy/thermodynamics layer (kept separate from static geometry)
