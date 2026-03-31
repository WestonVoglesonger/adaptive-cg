"""Download molecular structures from PDB/Zenodo."""
from pathlib import Path

CATEGORIES = {
    "proteins": [
        # Small globular (<100 residues)
        "1UBQ",  # ubiquitin, 76 res, mixed alpha/beta
        "1LYZ",  # lysozyme, 129 res, alpha+beta
        "5PTI",  # BPTI, 58 res, beta-rich
        "1CRN",  # crambin, 46 res, alpha-helical
        "1VII",  # villin headpiece, 36 res, three-helix bundle
        "2LZM",  # T4 lysozyme, 164 res, domain structure
        "1IGD",  # immunoglobulin-binding domain, 61 res, all-beta
        "1PGA",  # protein G B1 domain, 56 res, alpha+beta
        "2GB1",  # protein G B1 (NMR), 56 res
        # Medium globular (100-300 residues)
        "4AKE",  # adenylate kinase, 214 res, multi-domain
        "1MBN",  # myoglobin, 153 res, all-alpha
        "2RN2",  # ribonuclease, 155 res, alpha+beta
        "3LZM",  # T4 lysozyme mutant, 164 res
        "1HHO",  # hemoglobin, 141 res/chain, all-alpha
        "2CGA",  # chymotrypsinogen A, 245 res, all-beta
        # Large / multi-domain (300+ residues)
        "1F6M",  # TBP-associated factor, 389 res, alpha solenoid
        "1AON",  # GroEL subunit, 524 res, multi-domain
        "3PQR",  # pyruvate kinase, 530 res, TIM barrel + domains
        # Membrane protein
        "1OCC",  # cytochrome c oxidase, membrane alpha-helical bundle
        # All-beta structures
        "1TEN",  # tenascin, 89 res, fibronectin III fold
        "7RSA",  # ribonuclease A, 124 res, classic alpha+beta
    ],
    "peptides": [
        "1L2Y",  # trp-cage, 20 res, miniprotein fold
        "2EVQ",  # beta-hairpin peptide
        "1LE1",  # alpha-helical peptide
        "1WN8",  # polyalanine alpha-helix, 10 res
        "2I9M",  # chignolin, 10 res, beta-hairpin
        "1UAO",  # polyproline II helix
        "2JOF",  # WW domain, 34 res, all-beta miniprotein
        "1RIJ",  # melittin, 26 res, amphipathic alpha-helix
    ],
    "nucleic_acids": [
        # DNA
        "1BNA",  # B-DNA dodecamer (Dickerson-Drew)
        "4C64",  # DNA 16-mer duplex
        "1D49",  # A-DNA octamer
        "1ZF1",  # Z-DNA hexamer, left-handed
        # RNA
        "1RNA",  # tRNA, cloverleaf structure
        "1EHZ",  # tRNA-Phe, high resolution
        "4TNA",  # tRNA crystal form
        "1DUH",  # RNA hairpin loop
        "2GDI",  # RNA duplex
    ],
    "ligand_complexes": [
        # PDB entries where the protein binds a notable small-molecule ligand.
        # Tests pipeline handling of HETATM records and mixed residue types.
        "1HHB",  # hemoglobin with heme (HEM)
        "3HYD",  # lipase with inhibitor
        "1HSG",  # HIV-1 protease with drug (indinavir)
        "3PTB",  # trypsin with benzamidine inhibitor
    ],
}

DATA_DIR = Path("data/structures")


def setup_parser(parser):
    parser.add_argument(
        "--category",
        choices=list(CATEGORIES.keys()) + ["all"],
        default="all",
        help="Category of molecules to fetch (default: all)",
    )
    parser.add_argument(
        "--pdb-ids",
        nargs="+",
        help="Specific PDB IDs to fetch (overrides --category)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Output directory (default: {DATA_DIR})",
    )


def execute(args):
    from Bio.PDB import PDBList

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.pdb_ids:
        pdb_ids = args.pdb_ids
    elif args.category == "all":
        pdb_ids = [pid for ids in CATEGORIES.values() for pid in ids]
    else:
        pdb_ids = CATEGORIES[args.category]

    print(f"Fetching {len(pdb_ids)} structures to {data_dir}/")

    fetcher = PDBList(verbose=False)
    fetched, skipped = 0, 0

    for pdb_id in pdb_ids:
        out = data_dir / f"{pdb_id.upper()}.pdb"
        if out.exists():
            print(f"  {pdb_id}: already exists, skipping")
            skipped += 1
            continue

        path = fetcher.retrieve_pdb_file(
            pdb_id,
            pdir=str(data_dir),
            file_format="pdb",
        )

        if path and Path(path).exists():
            Path(path).rename(out)
            print(f"  {pdb_id}: downloaded")
            fetched += 1
        else:
            print(f"  {pdb_id}: FAILED")

    print(f"\nDone: {fetched} fetched, {skipped} skipped")
    return 0
