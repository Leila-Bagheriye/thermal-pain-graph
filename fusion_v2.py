"""
Fusion of the wearable (EDA) and contactless (video) branches, leave-one-subject-out.

Inputs
  --bio_feats  feats_BL1_PA4.csv written by train_bio_loso.py (columns: subject, label, file, eda_*)
  --video_run  a finished run folder with fold_*.npz (keys, y, prob) from stgraph_v2.py or finetune_v2.py
               (every video probability there is already out-of-subject)

Rules (all evaluated per held-out participant)
  eda      EDA logistic regression alone (reference)
  video    the saved video probability alone
  mean     (p_eda + p_video) / 2                 no learned weights -> no leakage
  stacked  logistic regression on (p_eda, p_video) trained on the other participants
           (p_eda for training trials from an inner grouped CV)

Run:
  python fusion_v2.py --bio_feats ../biovid_bio/feats_BL1_PA4.csv --video_run runs/graph --out results_v2/fusion_graph.json
"""
import os, glob, json, argparse
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

p = argparse.ArgumentParser()
p.add_argument("--bio_feats", required=True)
p.add_argument("--video_run", required=True)
p.add_argument("--out", default="fusion.json")
a = p.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

# ---- video probabilities (out-of-subject) ----
K, PV = [], []
for f in sorted(glob.glob(os.path.join(a.video_run, "fold_*.npz"))):
    z = np.load(f, allow_pickle=True)
    K += [str(k) for k in z["keys"]]; PV += list(z["prob"])
vid = dict(zip(K, PV))
print("video trials:", len(vid))

# ---- EDA features ----
B = pd.read_csv(a.bio_feats)
B["key"] = B.file.str.replace("_bio.csv", "", regex=False).str.replace(".csv", "", regex=False)
D = B[B.key.isin(vid)].reset_index(drop=True)
print("matched trials:", len(D), " subjects:", D.subject.nunique())
eda = [c for c in D.columns if c.startswith("eda_")]
X = D[eda].fillna(D[eda].median())
X = X.groupby(D.subject).transform(lambda s: (s - s.mean()) / (s.std() + 1e-8)).values  # per participant, no labels
y, g = D.label.values, D.subject.values
pv = np.array([vid[k] for k in D.key])
eda_m = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))

res = {k: [] for k in ["eda", "video", "mean", "stacked"]}
pred_all = {k: np.zeros(len(y), int) for k in res}
W = []
for s in np.unique(g):
    tr, te = g != s, g == s
    pe_tr = cross_val_predict(eda_m(), X[tr], y[tr], groups=g[tr], cv=GroupKFold(5), method="predict_proba")[:, 1]
    pe_te = eda_m().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    meta = LogisticRegression().fit(np.c_[pe_tr, pv[tr]], y[tr])
    W.append(meta.coef_[0].tolist())
    preds = {"eda": pe_te > .5, "video": pv[te] > .5, "mean": (pe_te + pv[te]) / 2 > .5,
             "stacked": meta.predict(np.c_[pe_te, pv[te]]).astype(bool)}
    for k, pr in preds.items():
        pr = pr.astype(int); pred_all[k][te] = pr
        res[k].append((accuracy_score(y[te], pr), f1_score(y[te], pr, average="macro")))
    print(f"{s}: " + "  ".join(f"{k}={res[k][-1][0]:.3f}" for k in res), flush=True)

out = {"video_run": a.video_run, "trials": int(len(y)), "subjects": int(len(np.unique(g)))}
for k, v in res.items():
    v = np.array(v)
    out[k] = {"acc_mean": float(v[:, 0].mean()), "acc_std": float(v[:, 0].std()),
              "f1_mean": float(v[:, 1].mean()), "pooled_acc": float(accuracy_score(y, pred_all[k])),
              "confusion_[[TN,FP],[FN,TP]]": confusion_matrix(y, pred_all[k]).tolist()}
    print(f"{k:8s} LOSO acc = {v[:,0].mean():.4f} ± {v[:,0].std():.4f}  F1 = {v[:,1].mean():.4f}")
# paired comparison with EDA alone, per participant
for k in ("mean", "stacked", "video"):
    d = np.array([r[0] for r in res[k]]) - np.array([r[0] for r in res["eda"]])
    out[k]["diff_vs_eda_mean"] = float(d.mean())
    out[k]["participants_better_same_worse"] = [int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())]
W = np.array(W)
out["stacked_weights_mean_[eda,video]"] = W.mean(0).tolist()
out["stacked_weights_std_[eda,video]"] = W.std(0).tolist()
print("stacked weights (EDA, video):", W.mean(0).round(3), "±", W.std(0).round(3))
json.dump(out, open(a.out, "w"), indent=2)
print("saved", a.out)
