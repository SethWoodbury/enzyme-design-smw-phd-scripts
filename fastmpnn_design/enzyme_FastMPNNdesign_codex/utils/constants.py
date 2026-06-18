"""Constants for Remastered FastMPNNdesign."""

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

HBOND_DISTANCE_MAX = 3.5
METAL_DISTANCE_MAX = 2.8
ELECTROSTATIC_DISTANCE_MAX = 4.5
HYDROPHOBIC_DISTANCE_MAX = 4.0
PI_DISTANCE_MAX = 5.0
COVALENT_DISTANCE_MAX = 2.0

AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
CHARGED_RESIDUES = {"ASP", "GLU", "LYS", "ARG", "HIS"}
NONPOLAR_RESIDUES = {
    "ALA",
    "VAL",
    "LEU",
    "ILE",
    "PRO",
    "MET",
    "PHE",
    "TRP",
}

DEFAULT_PROBABILITY_OF_MUTATION = 0.0

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
