"""
Load and classify molecular structures from PDB files.

Provides MoleculeData for the CG pipeline: heavy-atom coordinates (in nm),
masses, element types, and per-atom region labels derived from secondary
structure (proteins) or chemical role (nucleic acids).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Residue-name constants
# ---------------------------------------------------------------------------
STANDARD_AMINO_ACIDS = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # common modified / protonation variants in PDB files
    "HID", "HIE", "HIP", "CYX", "MSE",
})

NUCLEIC_ACID_RESNAMES = frozenset({
    # DNA
    "DA", "DT", "DG", "DC",
    # RNA
    "A", "U", "G", "C",
    # longer forms sometimes seen
    "ADE", "THY", "GUA", "CYT", "URA",
})

WATER_RESNAMES = frozenset({"HOH", "WAT", "SOL", "TIP3", "TIP4", "TIP5", "SPC"})
ION_RESNAMES = frozenset({
    "NA", "CL", "K", "MG", "CA", "ZN", "FE", "MN", "CU", "CO",
    "NI", "CD", "SE", "BR", "IOD", "LI", "RB", "CS", "BA", "SR",
    "NA+", "CL-", "K+", "MG2", "CA2", "ZN2",
})

# Nucleic-acid atom-name classification
_PHOSPHATE_ATOMS = frozenset({
    "P", "OP1", "OP2", "OP3", "O5'", "O3'",
    # PDB sometimes drops the prime
    "O5*", "O3*",
})
_SUGAR_ATOMS = frozenset({
    "C1'", "C2'", "C3'", "C4'", "C5'", "O4'", "O2'",
    "C1*", "C2*", "C3*", "C4*", "C5*", "O4*", "O2*",
})

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class MoleculeData:
    """Processed molecule ready for coarse-graining."""
    name: str                                  # PDB ID
    n_atoms: int                               # heavy atoms only
    positions: np.ndarray                      # (N, 3) nanometers
    masses: np.ndarray                         # (N,) daltons
    elements: list[str]                        # (N,)
    region_labels: np.ndarray                  # (N,) int
    region_names: list[str]                    # human-readable per index
    mol_type: str                              # protein | nucleic_acid | small_molecule

    def __repr__(self) -> str:
        regions = ", ".join(
            f"{name}={int((self.region_labels == i).sum())}"
            for i, name in enumerate(self.region_names)
        )
        return (
            f"MoleculeData({self.name}, {self.mol_type}, "
            f"n_atoms={self.n_atoms}, regions=[{regions}])"
        )


# ---------------------------------------------------------------------------
# Molecule-type detection
# ---------------------------------------------------------------------------

def _detect_mol_type(residue_names: list[str]) -> str:
    """Classify molecule type from the set of residue names present."""
    resname_set = {r.strip() for r in residue_names}
    n_protein = sum(1 for r in residue_names if r.strip() in STANDARD_AMINO_ACIDS)
    n_nucleic = sum(1 for r in residue_names if r.strip() in NUCLEIC_ACID_RESNAMES)
    total = len(residue_names)

    if total == 0:
        return "small_molecule"

    # Majority vote
    if n_protein / total > 0.5:
        return "protein"
    if n_nucleic / total > 0.5:
        return "nucleic_acid"
    # If there's a meaningful protein presence (e.g. complexes), still call it protein
    if n_protein > 0 and n_protein >= n_nucleic:
        return "protein"
    if n_nucleic > 0:
        return "nucleic_acid"
    return "small_molecule"


# ---------------------------------------------------------------------------
# Region labelling
# ---------------------------------------------------------------------------

def _label_protein_regions(universe, atom_indices) -> tuple[np.ndarray, list[str]]:
    """Assign helix / sheet / loop labels to protein heavy atoms.

    Tries DSSP via MDAnalysis first.  Falls back to labelling everything
    as 'loop' if DSSP is unavailable or fails.
    """
    region_names = ["helix", "sheet", "loop"]
    dssp_to_region = {
        "H": 0, "G": 0, "I": 0,   # helix family
        "E": 1, "B": 1,            # sheet family
    }

    ag = universe.atoms[atom_indices]
    n = len(ag)
    labels = np.full(n, 2, dtype=np.int32)  # default = loop

    try:
        from MDAnalysis.analysis.dssp import DSSP
        dssp = DSSP(universe).run()
        # dssp.results.dssp gives per-residue assignment as a string for each frame
        # We use frame 0 (only frame for a PDB).
        dssp_string = dssp.results.dssp[0]  # one char per residue in the whole universe

        # Build residue-index -> DSSP code mapping for residues in selection
        all_resids = list(universe.residues.resids)
        all_segids = list(universe.residues.segids)

        # Map (segid, resid) -> dssp code
        res_dssp = {}
        for idx, res in enumerate(universe.residues):
            if idx < len(dssp_string):
                res_dssp[(res.segid, res.resid)] = dssp_string[idx]

        for i, atom in enumerate(ag):
            code = res_dssp.get((atom.segid, atom.resid), "-")
            labels[i] = dssp_to_region.get(code, 2)

    except Exception:
        # DSSP not available or failed -- everything stays as loop (2)
        pass

    return labels, region_names


def _label_nucleic_regions(atom_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Classify nucleic-acid atoms into phosphate / sugar / base."""
    region_names = ["base", "sugar", "phosphate"]
    labels = np.zeros(len(atom_names), dtype=np.int32)

    for i, name in enumerate(atom_names):
        stripped = name.strip()
        if stripped in _PHOSPHATE_ATOMS:
            labels[i] = 2  # phosphate
        elif stripped in _SUGAR_ATOMS:
            labels[i] = 1  # sugar
        else:
            labels[i] = 0  # base
    return labels, region_names


def _label_small_molecule(n: int) -> tuple[np.ndarray, list[str]]:
    """All atoms belong to a single region."""
    return np.zeros(n, dtype=np.int32), ["molecule"]


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_molecule(pdb_path: Path) -> MoleculeData:
    """Load a PDB file and return a MoleculeData instance.

    - Selects heavy atoms only (no H), first chain, first altloc
    - Excludes water and ions
    - Converts positions from Angstroms to nanometers
    - Assigns per-atom region labels based on molecule type
    """
    import MDAnalysis as mda

    pdb_path = Path(pdb_path)
    pdb_id = pdb_path.stem.upper()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(str(pdb_path))

    # ------------------------------------------------------------------
    # Build selection: heavy atoms, first chain, no water/ions
    # ------------------------------------------------------------------
    # Collect resnames to exclude
    exclude_resnames = WATER_RESNAMES | ION_RESNAMES
    exclude_list = " ".join(sorted(exclude_resnames))

    # Start with all heavy atoms, exclude water/ions
    sel_string = f"not type H and not resname {exclude_list}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ag = u.select_atoms(sel_string)

    if len(ag) == 0:
        raise ValueError(f"No heavy atoms found in {pdb_path}")

    # First chain only
    chains = sorted(set(ag.segids))
    if len(chains) > 1:
        first_chain = chains[0]
        ag = ag.select_atoms(f"segid {first_chain}")

    # Handle alternate conformations: keep only first altloc (A or '')
    if hasattr(ag, "altLocs"):
        try:
            altlocs = ag.altLocs
            unique_alts = set(altlocs)
            # Keep atoms with no altloc ('', ' ') or altloc 'A'
            if len(unique_alts - {"", " "}) > 0:
                keep_mask = np.array(
                    [(a in ("", " ", "A")) for a in altlocs], dtype=bool
                )
                # For atoms sharing the same (resid, name), keep only one
                seen = set()
                final_mask = np.zeros(len(ag), dtype=bool)
                for i, atom in enumerate(ag):
                    key = (atom.segid, atom.resid, atom.name)
                    if keep_mask[i] and key not in seen:
                        final_mask[i] = True
                        seen.add(key)
                    elif key not in seen and not keep_mask[i]:
                        # No preferred altloc yet -- take whatever is first
                        final_mask[i] = True
                        seen.add(key)
                ag = ag[final_mask]
        except Exception:
            pass  # altLoc attribute not populated -- fine, skip

    if len(ag) == 0:
        raise ValueError(f"No atoms remaining after filtering in {pdb_path}")

    # ------------------------------------------------------------------
    # Extract arrays
    # ------------------------------------------------------------------
    positions = ag.positions / 10.0  # Angstroms -> nm
    masses = ag.masses.copy()

    # Filter out atoms with zero mass (e.g. unknown elements in some PDBs).
    nonzero = masses > 0
    if not nonzero.all():
        n_bad = (~nonzero).sum()
        positions = positions[nonzero]
        masses = masses[nonzero]
        # Rebuild ag for downstream use (residue names, atom names, etc.)
        ag = ag[nonzero]
        if len(ag) == 0:
            raise ValueError(f"All atoms have zero mass in {pdb_path}")
    elements = [atom.element.strip() if atom.element else atom.name.strip()[0]
                for atom in ag]
    residue_names = [atom.resname.strip() for atom in ag]
    atom_names = [atom.name.strip() for atom in ag]

    # ------------------------------------------------------------------
    # Detect molecule type and assign regions
    # ------------------------------------------------------------------
    mol_type = _detect_mol_type(residue_names)

    if mol_type == "protein":
        region_labels, region_names = _label_protein_regions(
            u, ag.indices
        )
    elif mol_type == "nucleic_acid":
        region_labels, region_names = _label_nucleic_regions(atom_names)
    else:
        region_labels, region_names = _label_small_molecule(len(ag))

    return MoleculeData(
        name=pdb_id,
        n_atoms=len(ag),
        positions=positions,
        masses=masses,
        elements=elements,
        region_labels=region_labels,
        region_names=region_names,
        mol_type=mol_type,
    )
