from workflow.ligand_based.methods.usrcat import USRCATMethod

# Map config method name → LigandBasedMethod class.
# To add a new method: create workflow/ligand_based/methods/<name>.py,
# implement LigandBasedMethod, then add an entry here.
METHODS = {
    "usrcat": USRCATMethod,
    "usr": USRCATMethod,  # same class; cfg controls which descriptor file is loaded
}


def get_method(name: str):
    if name not in METHODS:
        raise ValueError(f"Unknown ligand-based method: {name!r}. Available: {list(METHODS)}")
    return METHODS[name]()
