# File: make_metrics.py
# Usage: python make_metrics.py --root LIT_PCBA --labels optional_labels.csv --out metrics_summary.csv
import os, re, csv, argparse, math, glob
from collections import defaultdict

def read_labels(path):
    labels = {}
    if not path: return labels
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        key = 'unique_id'
        if key not in r.fieldnames:
            raise SystemExit(f"labels file needs 'unique_id' column, got {r.fieldnames}")
        for row in r:
            uid = row['unique_id']
            val = row.get('pK') or row.get('label') or ''
            labels[uid] = val
    return labels

def parse_vina_affinity_from_pdbqt(pdbqt_path):
    # Look for: "REMARK VINA RESULT:  -8.5  0.000  0.000"
    try:
        with open(pdbqt_path, 'r', errors='ignore') as f:
            for line in f:
                if line.startswith('REMARK VINA RESULT:'):
                    parts = line.strip().split()
                    return float(parts[3])  # affinity
    except Exception:
        pass
    return None

def spearman(xs, ys):
    # simple Spearman rho (with average ranks for ties)
    def ranks(a):
        sorted_idx = sorted(range(len(a)), key=lambda i: a[i])
        r = [0]*len(a); i=0
        while i < len(a):
            j=i
            while j+1 < len(a) and a[sorted_idx[j+1]] == a[sorted_idx[i]]:
                j+=1
            rank = 0.5*(i+j) + 1
            for k in range(i, j+1):
                r[sorted_idx[k]] = rank
            i=j+1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx)/len(rx); my = sum(ry)/len(ry)
    cov = sum((rx[i]-mx)*(ry[i]-my) for i in range(len(rx)))
    sx = math.sqrt(sum((v-mx)**2 for v in rx))
    sy = math.sqrt(sum((v-my)**2 for v in ry))
    return cov/(sx*sy) if sx and sy else float('nan')

def roc_auc(labels, scores):
    # labels: 1=active, 0=inactive; scores: higher = better hit
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    pos = sum(labels); neg = len(labels)-pos
    if pos==0 or neg==0: return float('nan')
    tp=fp=0; prev_s=None; auc=0.0; last_tp=last_fp=0
    for s, y in pairs:
        if s!=prev_s:
            auc += (fp-last_fp) * (tp+last_tp)/2.0
            prev_s=s; last_tp=tp; last_fp=fp
        if y==1: tp+=1
        else: fp+=1
    auc += (fp-last_fp) * (tp+last_tp)/2.0
    return auc/(pos*neg)

def enrichment_at(labels, scores, frac=0.01):
    # fraction of dataset selected; return EF@frac wrt random
    n = len(labels); k = max(1, int(n*frac))
    idx = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    hits = sum(labels[i] for i in idx)
    total_actives = sum(labels)
    if total_actives==0: return float('nan')
    expected = k * (total_actives/n)
    return hits/expected if expected>0 else float('nan')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--labels', default='')
    ap.add_argument('--out', default='metrics_summary.csv')
    args = ap.parse_args()

    labels_map = read_labels(args.labels)

    rows_out = []
    metrics = []

    # Iterate all per-protein predictions
    pred_paths = sorted(glob.glob(os.path.join(args.root, '*', 'rescoring', 'results', 'aev_plig_*_*.csv')))
    if not pred_paths:
        raise SystemExit("No prediction CSVs found under */rescoring/results/")

    for pred_csv in pred_paths:
        # Infer target/mode from filename
        base = os.path.basename(pred_csv)  # aev_plig_<TARGET>_<MODE>_preds.csv
        parts = base.split('_')
        target = parts[2]
        mode = parts[3]
        protein_dir = os.path.join(args.root, target)

        # Collect predictions
        preds = {}
        with open(pred_csv, newline='') as f:
            r = csv.DictReader(f)
            if 'unique_id' not in r.fieldnames:
                raise SystemExit(f"{pred_csv} missing 'unique_id'")
            # Allow score column to be named a few common variants
            score_col = None
            for cand in ('aev_plig_score','score','pred','prediction'):
                if cand in r.fieldnames:
                    score_col = cand; break
            if not score_col:
                raise SystemExit(f"{pred_csv} missing score column")
            for row in r:
                preds[row['unique_id']] = float(row[score_col])

        # Walk docked outputs to fetch Vina affinity
        affins = {}  # unique_id -> affinity
        dock_dir = os.path.join(protein_dir, 'docked_vina', mode)
        for pdbqt in glob.glob(os.path.join(dock_dir, '*.pdbqt')):
            uid = os.path.splitext(os.path.basename(pdbqt))[0]
            aff = parse_vina_affinity_from_pdbqt(pdbqt)
            if aff is not None:
                affins[uid] = aff

        # Merge per ligand
        for uid, score in preds.items():
            row = {
                'target': target,
                'mode': mode,
                'unique_id': uid,
                'aev_plig_score': score,
                'vina_affinity': affins.get(uid, ''),
                'label_pK': labels_map.get(uid, '')
            }
            rows_out.append(row)

        # Compute per-(target,mode) metrics if we can label actives
        # Use mode as proxy if no label file: treat "actives" as positives
        ys = []
        s_plig = []
        s_vina = []
        for uid, score in preds.items():
            if args.labels:
                lab = labels_map.get(uid, '')
                if lab == '': continue
                try:
                    pk = float(lab)
                    # Consider higher pK = more active; binarize at median? leave metrics below numeric
                    ys.append(1 if pk>=7.0 else 0)  # simple threshold; tweak as needed
                except:
                    ys.append(1 if str(lab).lower() in ('1','true','active') else 0)
            else:
                ys.append(1 if mode=='actives' else 0)
            s_plig.append(score)
            va = affins.get(uid)
            if va is not None:
                s_vina.append(-va)  # lower (more negative) affinity is better ? invert for ranking
            else:
                s_vina.append(float('-inf'))

        if ys:
            auc_plig = roc_auc(ys, s_plig)
            ef1 = enrichment_at(ys, s_plig, 0.01)
            ef5 = enrichment_at(ys, s_plig, 0.05)
            ef10= enrichment_at(ys, s_plig, 0.10)
            # correlation between methods where both exist
            both = [(s_plig[i], s_vina[i]) for i in range(len(ys)) if math.isfinite(s_vina[i])]
            spr = float('nan')
            if both:
                spr = spearman([b[0] for b in both], [b[1] for b in both])
            metrics.append({
                'target': target, 'mode': mode,
                'n': len(ys),
                'roc_auc_aev_plig': auc_plig,
                'ef1%_aev_plig': ef1, 'ef5%_aev_plig': ef5, 'ef10%_aev_plig': ef10,
                'spearman_plig_vs_vina': spr
            })

    # Write merged per-ligand table (for downstream analysis)
    merged_csv = os.path.splitext(args.out)[0] + "_per_ligand.csv"
    if rows_out:
        keys = ['target','mode','unique_id','aev_plig_score','vina_affinity','label_pK']
        with open(merged_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows_out)
        print(f"Wrote {merged_csv}  ({len(rows_out)} rows)")

    # Write metrics summary
    if metrics:
        keys = ['target','mode','n','roc_auc_aev_plig','ef1%_aev_plig','ef5%_aev_plig','ef10%_aev_plig','spearman_plig_vs_vina']
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(metrics)
        print(f"Wrote {args.out}  ({len(metrics)} rows)")
    else:
        print("No metrics computed (no labels and no mode inference?)")

if __name__ == '__main__':
    main()
