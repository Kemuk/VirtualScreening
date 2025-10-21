from pathlib import Path
import pandas as pd

def read_vina_boxes(path):
    """
    Returns dict target_id -> dict of box/template info.
    Expects columns: target_id, template_ligand, box_center_x,y,z, box_size_x,y,z (best-effort).
    """
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        tid = str(r.get("target_id") or r.get("target") or r.get("id"))
        out.setdefault(tid, {})
        out[tid]["template_ligand"] = r.get("template_ligand")
        out[tid]["box"] = {
            "center": (r.get("box_center_x"), r.get("box_center_y"), r.get("box_center_z")),
            "size": (r.get("box_size_x"), r.get("box_size_y"), r.get("box_size_z"))
        }
    return out

def list_targets(data_root):
    data_root = Path(data_root)
    return [p.name for p in data_root.iterdir() if p.is_dir()]
