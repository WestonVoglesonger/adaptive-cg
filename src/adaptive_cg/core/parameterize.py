"""CG force field parameterization via Boltzmann inversion.

Pools extracted distributions across molecules by bead-class pair,
then applies U(r) = -kT ln P(r) to derive bonded potentials. Fits
harmonic approximations for bonds and angles.

Produces a transferable CG force field: given any bead's chemical
class, look up its interaction parameters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


# Boltzmann constant in kJ/(mol·K)
KB = 0.008314462618


@dataclass
class HarmonicParams:
    """Parameters for a harmonic potential U = k(x - x0)^2."""
    x0: float      # equilibrium value (nm for bonds, rad for angles)
    k: float       # force constant (kJ/mol/nm² for bonds, kJ/mol/rad² for angles)
    n_samples: int  # number of samples used to fit


@dataclass
class CGForceField:
    """Transferable CG force field parameters."""

    # Bond params keyed by "classA--classB"
    bond_params: dict[str, HarmonicParams]

    # Angle params keyed by "classA--classB--classC"
    angle_params: dict[str, HarmonicParams]

    # Non-bonded LJ params keyed by "classA--classB"
    # Each has (sigma, epsilon) — sigma in nm, epsilon in kJ/mol
    nonbonded_params: dict[str, tuple[float, float]]

    # Temperature used for Boltzmann inversion
    temperature: float

    # Source molecules used in parameterization
    source_molecules: list[str]

    def save(self, path: Path):
        """Save force field to JSON."""
        data = {
            "temperature": self.temperature,
            "source_molecules": self.source_molecules,
            "bonds": {
                k: {"x0": v.x0, "k": v.k, "n_samples": v.n_samples}
                for k, v in self.bond_params.items()
            },
            "angles": {
                k: {"x0": v.x0, "k": v.k, "n_samples": v.n_samples}
                for k, v in self.angle_params.items()
            },
            "nonbonded": {
                k: {"sigma": v[0], "epsilon": v[1]}
                for k, v in self.nonbonded_params.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: Path) -> "CGForceField":
        """Load force field from JSON."""
        with open(path) as f:
            data = json.load(f)
        return CGForceField(
            bond_params={
                k: HarmonicParams(v["x0"], v["k"], v["n_samples"])
                for k, v in data["bonds"].items()
            },
            angle_params={
                k: HarmonicParams(v["x0"], v["k"], v["n_samples"])
                for k, v in data["angles"].items()
            },
            nonbonded_params={
                k: (v["sigma"], v["epsilon"])
                for k, v in data["nonbonded"].items()
            },
            temperature=data["temperature"],
            source_molecules=data["source_molecules"],
        )


def boltzmann_invert_harmonic(
    samples: np.ndarray,
    temperature: float,
    n_bins: int = 100,
) -> HarmonicParams:
    """Derive harmonic parameters from a distribution via Boltzmann inversion.

    1. Histogram the samples
    2. U(x) = -kT ln P(x)
    3. Fit a harmonic U = k(x - x0)^2 to the PMF

    Parameters
    ----------
    samples : np.ndarray
        Distribution samples (bond lengths in nm, angles in rad).
    temperature : float
        Temperature in Kelvin.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    HarmonicParams
    """
    kbt = KB * temperature
    n_samples = len(samples)

    if n_samples < 10:
        # Too few samples — use simple mean/variance estimate
        x0 = float(np.mean(samples))
        variance = float(np.var(samples))
        k = kbt / variance if variance > 1e-12 else 1000.0
        return HarmonicParams(x0=x0, k=k, n_samples=n_samples)

    # Histogram
    counts, bin_edges = np.histogram(samples, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Remove zero-count bins (can't take log)
    mask = counts > 0
    x = bin_centers[mask]
    p = counts[mask]

    # PMF via Boltzmann inversion
    pmf = -kbt * np.log(p)
    pmf -= pmf.min()  # shift minimum to zero

    # Fit harmonic: U(x) = k * (x - x0)^2
    def harmonic(x, k, x0):
        return k * (x - x0) ** 2

    # Fallback values from distribution statistics
    x0_fallback = float(np.mean(samples))
    variance = float(np.var(samples))
    k_fallback = kbt / variance if variance > 1e-12 else 1000.0

    try:
        # Initial guess: x0 = peak of distribution, k from variance
        x0_guess = x[np.argmin(pmf)]
        k_guess = kbt / (variance + 1e-12)
        popt, _ = curve_fit(harmonic, x, pmf, p0=[k_guess, x0_guess], maxfev=5000)
        k_fit, x0_fit = float(abs(popt[0])), float(popt[1])

        # Sanity check: reject nonsense fits
        sample_min, sample_max = float(np.min(samples)), float(np.max(samples))
        if x0_fit < sample_min - (sample_max - sample_min) or \
           x0_fit > sample_max + (sample_max - sample_min) or \
           k_fit < 1e-3:
            x0_fit, k_fit = x0_fallback, k_fallback
    except (RuntimeError, ValueError):
        x0_fit, k_fit = x0_fallback, k_fallback

    return HarmonicParams(x0=x0_fit, k=k_fit, n_samples=n_samples)


def derive_lj_from_rdf(
    distances: np.ndarray,
    temperature: float,
    n_bins: int = 80,
) -> tuple[float, float]:
    """Estimate LJ sigma and epsilon from non-bonded distance distribution.

    sigma ≈ distance at which PMF crosses zero (first repulsive wall).
    epsilon ≈ depth of the PMF minimum.

    Returns (sigma, epsilon) in (nm, kJ/mol).
    """
    kbt = KB * temperature

    if len(distances) < 20:
        # Not enough data — return default soft repulsion
        return (0.4, 1.0)

    counts, bin_edges = np.histogram(distances, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mask = counts > 0
    x = bin_centers[mask]
    p = counts[mask]

    pmf = -kbt * np.log(p)
    pmf -= pmf.min()

    # Find minimum of PMF (equilibrium distance)
    min_idx = np.argmin(pmf)
    r_min = float(x[min_idx])
    epsilon = float(pmf.max() - pmf.min()) if len(pmf) > 1 else 1.0

    # sigma ≈ r_min / 2^(1/6) (LJ relation between sigma and r_min)
    sigma = r_min / (2.0 ** (1.0 / 6.0)) if r_min > 0.1 else 0.3

    # Clamp to reasonable ranges
    sigma = np.clip(sigma, 0.2, 1.5)
    epsilon = np.clip(epsilon, 0.1, 20.0)

    return (float(sigma), float(epsilon))


def parameterize_forcefield(
    extracted_dirs: list[Path],
    temperature: float = 300.0,
    verbose: bool = True,
) -> CGForceField:
    """Build a transferable CG force field from extracted distributions.

    Pools distributions across molecules by bead-class pair, then
    applies Boltzmann inversion.

    Parameters
    ----------
    extracted_dirs : list[Path]
        Directories containing extraction results (from `acg extract`).
    temperature : float
        Temperature in Kelvin.
    verbose : bool
        Print progress.

    Returns
    -------
    CGForceField
    """
    from adaptive_cg.core.extract import ExtractionResult

    # Pool distributions across all molecules
    pooled_bonds: dict[str, list[float]] = {}
    pooled_angles: dict[str, list[float]] = {}
    pooled_nonbond: dict[str, list[float]] = {}
    source_molecules = []

    for d in extracted_dirs:
        if verbose:
            print(f"Loading: {d}")
        result = ExtractionResult.load(d)
        source_molecules.append(result.molecule)

        for k, v in result.bond_distributions.items():
            pooled_bonds.setdefault(k, []).extend(v)
        for k, v in result.angle_distributions.items():
            pooled_angles.setdefault(k, []).extend(v)
        for k, v in result.nonbonded_distances.items():
            pooled_nonbond.setdefault(k, []).extend(v)

    if verbose:
        print(f"\nPooled from {len(source_molecules)} molecules:")
        print(f"  Bond types: {len(pooled_bonds)}")
        print(f"  Angle types: {len(pooled_angles)}")
        print(f"  Non-bonded types: {len(pooled_nonbond)}")

    # --- Boltzmann inversion for bonds ---
    bond_params = {}
    if verbose:
        print("\nFitting bond potentials:")
    for key, samples in sorted(pooled_bonds.items()):
        arr = np.array(samples)
        params = boltzmann_invert_harmonic(arr, temperature)
        bond_params[key] = params
        if verbose:
            print(f"  {key}: r0={params.x0:.4f} nm, "
                  f"k={params.k:.1f} kJ/mol/nm², "
                  f"n={params.n_samples}")

    # --- Boltzmann inversion for angles ---
    angle_params = {}
    if verbose:
        print("\nFitting angle potentials:")
    for key, samples in sorted(pooled_angles.items()):
        arr = np.array(samples)
        params = boltzmann_invert_harmonic(arr, temperature)
        angle_params[key] = params
        if verbose:
            deg = np.degrees(params.x0)
            print(f"  {key}: theta0={deg:.1f} deg, "
                  f"k={params.k:.1f} kJ/mol/rad², "
                  f"n={params.n_samples}")

    # --- Non-bonded LJ from RDF ---
    nonbonded_params = {}
    if verbose:
        print("\nFitting non-bonded potentials:")
    for key, samples in sorted(pooled_nonbond.items()):
        arr = np.array(samples)
        sigma, epsilon = derive_lj_from_rdf(arr, temperature)
        nonbonded_params[key] = (sigma, epsilon)
        if verbose:
            print(f"  {key}: sigma={sigma:.4f} nm, "
                  f"eps={epsilon:.2f} kJ/mol, "
                  f"n={len(samples)}")

    ff = CGForceField(
        bond_params=bond_params,
        angle_params=angle_params,
        nonbonded_params=nonbonded_params,
        temperature=temperature,
        source_molecules=source_molecules,
    )

    if verbose:
        print(f"\nForce field: {len(bond_params)} bond types, "
              f"{len(angle_params)} angle types, "
              f"{len(nonbonded_params)} non-bonded types")

    return ff
