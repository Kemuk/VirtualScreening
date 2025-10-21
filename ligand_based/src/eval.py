import pandas as pd
from rdkit.ML.Scoring import Scoring
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_rankings(ranked_csv_path, labels_csv_path):
    """
    Use labels file as ground truth. Merge ranked on smiles, drop others.
    ranked_csv: must have columns 'smiles' and 'score'
    labels_csv: must have 'smiles' and a label column among known names
    Returns dict: {roc_auc, pr_auc, ef_1pct, bedroc_20} (floats or None)
    """
    ranked = pd.read_csv(ranked_csv_path)
    labels = pd.read_csv(labels_csv_path)

    # check columns
    if "smiles" not in ranked.columns or "score" not in ranked.columns:
        raise ValueError(f"Ranked file {ranked_csv_path} missing 'smiles' / 'score'. Columns: {ranked.columns}")
    if "smiles" not in labels.columns:
        raise ValueError(f"Labels file {labels_csv_path} missing 'smiles'. Columns: {labels.columns}")

    # Find the truth label column
    candidate_labels = ["label", "Label", "activity", "Activity", "y", "Y", "class", "Class"]
    label_col = None
    for c in candidate_labels:
        if c in labels.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"No label column found in {labels_csv_path}. Options tried: {candidate_labels}")

    # Normalize truth labels to 1 (active) / 0 (inactive)
    labels = labels.copy()
    lab_series = labels[label_col].astype(str).str.lower()
    labels["__y__"] = (lab_series == "active").astype(int)

    # Merge
    merged = ranked.merge(labels[["smiles", "__y__"]], on="smiles", how="inner")
    if merged.empty:
        raise ValueError(f"No overlap between ranked and labels (by smiles) in {ranked_csv_path}")

    # Extract scores and labels
    scores = merged["score"].astype(float).tolist()
    y = merged["__y__"].astype(int).tolist()

    # Compute simple metrics
    roc = None
    pr = None
    if len(set(y)) > 1:
        roc = float(roc_auc_score(y, scores))
        pr = float(average_precision_score(y, scores))

    # Prepare data for RDKit scoring: list of [score, label]
    data = [[scores[i], y[i]] for i in range(len(y))]
    # sort descending by score
    data = sorted(data, key=lambda x: x[0], reverse=True)

    # Enrichment and BEDROC
    ef_list = Scoring.CalcEnrichment(data, col=1, fractions=[0.01, 0.05,0.1])
    ef1, ef5, ef10 = ef_list
    bedroc20 = Scoring.CalcBEDROC(data, col=1, alpha=20.0)

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "ef_1pct": ef1,
        "ef_5pct": ef5,
        "ef_10pct": ef10,
        "bedroc_20": float(bedroc20)
    }
