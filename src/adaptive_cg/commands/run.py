"""Command: auto-configured adaptive CG simulation.

Detects hardware, picks optimal bead count, runs adaptive simulation
with compute-aware bead scaling and quality floor enforcement.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("molecule", help="PDB ID (e.g. 1UBQ)")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--target-rate", type=float, default=None,
                        help="Target steps/second (default: auto from hardware)")
    parser.add_argument("--max-beads", type=int, default=None)
    parser.add_argument("--min-beads", type=int, default=20)
    parser.add_argument("--quality-floor", type=float, default=0.5,
                        help="Minimum structural quality score 0-1")
    parser.add_argument("--n-regions", type=int, default=5)
    parser.add_argument("--forcefield", type=str, default="data/forcefield/cg_forcefield.json")
    parser.add_argument("--output-dir", type=str, default=None)


def execute(args: argparse.Namespace) -> int:
    import warnings

    import numpy as np

    from adaptive_cg.core.hardware import detect_hardware, estimate_max_beads
    from adaptive_cg.core.compute_budget import ComputeBudget, auto_configure
    from adaptive_cg.core.quality import compute_quality, meets_quality_floor
    from adaptive_cg.core.engine import (
        setup_cg_system, langevin_step, minimize_energy, KB,
    )

    mol_id = args.molecule.upper()
    data_dir = Path("data")
    pdb_path = data_dir / "structures" / f"{mol_id}.pdb"
    ff_path = Path(args.forcefield)

    if not pdb_path.exists():
        print(f"Error: PDB file not found: {pdb_path}")
        print(f"Run `acg fetch` first to download {mol_id}")
        return 1

    if not ff_path.exists():
        print(f"Error: Force field not found: {ff_path}")
        print("Run `acg parameterize` first")
        return 1

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "run_trajectories" / mol_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Detect hardware
    # ------------------------------------------------------------------
    print(f"=== Auto-configured CG Simulation: {mol_id} ===\n")
    print("Detecting hardware...")
    hw = detect_hardware()
    print(f"  CPU:    {hw.cpu_name} ({hw.cpu_cores} cores)")
    if hw.gpu_available:
        print(f"  GPU:    {hw.gpu_name}")
    else:
        print(f"  GPU:    none detected")
    print(f"  Memory: {hw.memory_gb:.1f} GB")
    print(f"  Throughput: {hw.estimated_pairs_per_second:.0f} pair evals/s")

    # ------------------------------------------------------------------
    # 2. Load atom data to determine system size
    # ------------------------------------------------------------------
    import MDAnalysis as mda
    from adaptive_cg.core.molecule import WATER_RESNAMES, ION_RESNAMES

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(str(pdb_path))
    exclude_list = " ".join(sorted(WATER_RESNAMES | ION_RESNAMES))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ag = u.select_atoms(f"not type H and not resname {exclude_list}")
    chains = sorted(set(ag.segids))
    if len(chains) > 1:
        ag = ag.select_atoms(f"segid {chains[0]}")
    nonzero = ag.masses > 0
    n_atoms = int(nonzero.sum())
    aa_positions = ag.positions[nonzero] / 10.0  # Angstroms -> nm

    print(f"\n  Molecule: {mol_id}, {n_atoms} heavy atoms")

    # ------------------------------------------------------------------
    # 3. Auto-configure compute budget
    # ------------------------------------------------------------------
    budget = auto_configure(
        n_atoms=n_atoms,
        target_steps_per_second=args.target_rate,
        hardware_pairs_per_second=hw.estimated_pairs_per_second,
    )

    # Apply user overrides
    if args.min_beads:
        budget.min_beads = max(budget.min_beads, args.min_beads)
    if args.max_beads:
        budget.max_beads = min(budget.max_beads, args.max_beads)
    budget.current_beads = max(budget.min_beads, min(budget.max_beads, budget.current_beads))

    initial_beads = budget.current_beads

    print(f"\nCompute budget:")
    print(f"  Target rate:   {budget.target_steps_per_second:.0f} steps/s")
    print(f"  Bead range:    [{budget.min_beads}, {budget.max_beads}]")
    print(f"  Initial beads: {initial_beads}")

    # ------------------------------------------------------------------
    # 4. Setup CG system
    # ------------------------------------------------------------------
    print(f"\nBuilding CG system with {initial_beads} beads...")
    system = setup_cg_system(
        pdb_path=pdb_path,
        ff_path=ff_path,
        n_beads=initial_beads,
        temperature=args.temperature,
        dihedral_scale=0.001,
        structure_bias="go",
        verbose=False,
    )
    print(f"  Beads: {system.n_beads}")
    print(f"  Topology: {len(system.bond_list)} bonds, "
          f"{len(system.angle_list)} angles, "
          f"{len(system.nb_pairs)} nb pairs")

    # ------------------------------------------------------------------
    # 5. Minimize energy
    # ------------------------------------------------------------------
    print("\nMinimizing energy...")
    minimize_energy(system, max_steps=5000, step_size=0.001, verbose=False)

    # Re-initialize velocities after minimization
    for i in range(system.n_beads):
        sigma_v = np.sqrt(KB * args.temperature / system.masses[i])
        system.velocities[i] = np.random.normal(0.0, sigma_v, size=3)
    com_vel = (
        (system.masses[:, None] * system.velocities).sum(axis=0)
        / system.masses.sum()
    )
    system.velocities -= com_vel

    # ------------------------------------------------------------------
    # 6. Run simulation loop with compute-aware bead scaling
    # ------------------------------------------------------------------
    total_time = args.steps * args.dt
    print(f"\nRunning: {args.steps} steps, dt={args.dt} ps "
          f"({total_time:.1f} ps total)")
    print(f"Quality floor: {args.quality_floor}")
    print()

    log_interval = 100
    budget_check_interval = budget.adjustment_interval
    quality_check_interval = 5000
    save_interval = 1000
    friction = 1.0

    forces, _ = system.compute_forces()
    frames = []
    bead_adjustments = []
    quality_history = []

    wall_start = time.perf_counter()

    for step in range(args.steps):
        step_start = time.perf_counter()

        # Integrate one step
        forces = langevin_step(system, forces, args.dt, args.temperature, friction)

        step_elapsed = time.perf_counter() - step_start
        budget.record_step(step_elapsed)

        # --- Budget check: adjust bead count if needed ---
        if budget.should_adjust(step):
            budget._last_adjustment_step = step
            recommended = budget.recommend_beads()

            if recommended != budget.current_beads:
                # Before reducing beads, check quality floor
                allow_change = True
                if recommended < budget.current_beads:
                    qm = compute_quality(
                        cg_positions=system.positions,
                        cg_trajectory=None,
                        aa_positions=aa_positions,
                        aa_trajectory=None,
                        n_regions=args.n_regions,
                    )
                    if not meets_quality_floor(qm, args.quality_floor):
                        allow_change = False
                        print(f"  Step {step+1}: budget recommends {recommended} beads "
                              f"but quality={qm.structural_quality:.3f} < "
                              f"floor={args.quality_floor} -- holding")

                if allow_change:
                    old_beads = budget.current_beads
                    budget.current_beads = recommended

                    # Rebuild system at new bead count
                    system = setup_cg_system(
                        pdb_path=pdb_path,
                        ff_path=ff_path,
                        n_beads=recommended,
                        temperature=args.temperature,
                        dihedral_scale=0.001,
                        structure_bias="go",
                        verbose=False,
                    )
                    minimize_energy(system, max_steps=2000, step_size=0.001, verbose=False)

                    # Re-init velocities
                    for i in range(system.n_beads):
                        sigma_v = np.sqrt(KB * args.temperature / system.masses[i])
                        system.velocities[i] = np.random.normal(0.0, sigma_v, size=3)
                    com_vel = (
                        (system.masses[:, None] * system.velocities).sum(axis=0)
                        / system.masses.sum()
                    )
                    system.velocities -= com_vel

                    forces, _ = system.compute_forces()
                    bead_adjustments.append((step + 1, old_beads, recommended))
                    print(f"  Step {step+1}: beads {old_beads} -> {recommended} "
                          f"(rate={budget.measured_rate:.1f} steps/s)")

        # --- Periodic quality check ---
        if (step + 1) % quality_check_interval == 0:
            qm = compute_quality(
                cg_positions=system.positions,
                cg_trajectory=None,
                aa_positions=aa_positions,
                aa_trajectory=None,
                n_regions=args.n_regions,
            )
            quality_history.append((step + 1, qm.structural_quality, qm.rg, qm.rg_reference))

        # --- Log energies ---
        if (step + 1) % (log_interval * 10) == 0:
            _, energies = system.compute_forces()
            temp = system.temperature()
            pe = energies["potential"]
            rate = budget.measured_rate
            elapsed = time.perf_counter() - wall_start
            print(f"  Step {step+1}/{args.steps}: T={temp:.1f} K, "
                  f"PE={pe:.1f} kJ/mol, beads={system.n_beads}, "
                  f"rate={rate:.1f} steps/s, wall={elapsed:.1f}s")

        # --- Save frame ---
        if (step + 1) % save_interval == 0:
            frames.append(system.positions.copy())

    wall_elapsed = time.perf_counter() - wall_start
    actual_rate = args.steps / wall_elapsed if wall_elapsed > 0 else 0.0

    # ------------------------------------------------------------------
    # 7. Final quality assessment
    # ------------------------------------------------------------------
    final_quality = compute_quality(
        cg_positions=system.positions,
        cg_trajectory=np.stack(frames) if len(frames) >= 2 else None,
        aa_positions=aa_positions,
        aa_trajectory=None,
        n_regions=args.n_regions,
    )

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    if frames:
        traj_path = output_dir / f"{mol_id}_run_traj.npy"
        np.save(traj_path, np.stack(frames))
    else:
        traj_path = None

    # ------------------------------------------------------------------
    # 9. Print summary
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  SUMMARY: {mol_id}")
    print(f"{'='*50}")
    print(f"  Total steps:     {args.steps}")
    print(f"  Wall time:       {wall_elapsed:.1f} s")
    print(f"  Avg rate:        {actual_rate:.1f} steps/s")
    print(f"  Target rate:     {budget.target_steps_per_second:.0f} steps/s")
    print(f"  Initial beads:   {initial_beads}")
    print(f"  Final beads:     {system.n_beads}")
    print(f"  Bead adjustments: {len(bead_adjustments)}")
    for step_num, old, new in bead_adjustments:
        print(f"    Step {step_num}: {old} -> {new}")
    print()
    print(f"  Quality (final):")
    print(f"    Structural quality: {final_quality.structural_quality:.3f}"
          f"  {'PASS' if meets_quality_floor(final_quality, args.quality_floor) else 'FAIL'}"
          f" (floor={args.quality_floor})")
    print(f"    Rg: {final_quality.rg:.3f} nm "
          f"(ref: {final_quality.rg_reference:.3f} nm, "
          f"dev: {final_quality.rg_deviation:.1%})")
    print(f"    Contact map corr: {final_quality.contact_map_correlation:.3f}")
    if final_quality.rmsf_correlation > 0:
        print(f"    RMSF corr: {final_quality.rmsf_correlation:.3f}")
    print()
    print(f"  Temperature:     {system.temperature():.1f} K")
    if traj_path:
        print(f"  Trajectory:      {traj_path}")
    print(f"  Output dir:      {output_dir}")

    return 0
