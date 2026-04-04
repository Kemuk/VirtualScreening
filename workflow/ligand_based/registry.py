from workflow.ligand_based.methods.fingerprint import ECFPMethod, MACCSMethod
from workflow.ligand_based.methods.usrcat import USRMethod, USRCATMethod

# Map config method name → LigandBasedMethod class.
# To add a new method: create workflow/ligand_based/methods/<name>.py,
# implement LigandBasedMethod, then add an entry here.
METHODS = {
    "ecfp4_tanimoto": ECFPMethod,
    "maccs_tanimoto": MACCSMethod,
    "usrcat": USRCATMethod,
    "usr": USRMethod,
}


def get_method(name: str):
    if name not in METHODS:
        raise ValueError(f"Unknown ligand-based method: {name!r}. Available: {list(METHODS)}")
    return METHODS[name]()
