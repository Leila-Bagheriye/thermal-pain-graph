"""
Wearable-pathway timing, reported the way Nienaber et al. do it:
feature computation per 5.5 s window, training time and testing time,
averaged over 10 participants. Run from the biovid_bio folder:

  python bio_latency.py --root . --out ../pain_v2/results_v2/bio_latency.json
"""
import os, glob, json, time, argparse, importlib.util
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

p = argparse.ArgumentParser()
p.add_argument("--root", default=".")
p.add_argument("--out", default="bio_latency.json")
p.add_argument("--n_subjects", type=int, default=10)
a = p.parse_args()

# reuse the exact feature function from train_bio_loso.py
src = open(os.path.join(a.root, "train_bio_loso.py")).read()
start = src.index("def eda_feats"); end = src.index("def ecg_feats")
ns = {"np": np, "linregress": __import__("scipy.stats", fromlist=["linregress"]).linregress, "FS": 512}
exec(src[start:end], ns); eda_feats = ns["eda_feats"]

files = sorted(f for f in glob.glob(os.path.join(a.root, "**", "*_bio.csv"), recursive=True)
               if "-BL1-" in f or "-PA4-" in f)
subj = sorted({os.path.basename(f).split("-")[0] for f in files})[:a.n_subjects]
win_t, read_t, X, y, g = [], [], [], [], []
for f in files:
    s = os.path.basename(f).split("-")[0]
    if s not in subj: continue
    t0 = time.perf_counter(); df = pd.read_csv(f, sep=None, engine="python"); read_t.append(time.perf_counter() - t0)
    df.columns = [c.strip().lower() for c in df.columns]
    x = df["gsr"].values.astype(float)
    t0 = time.perf_counter(); fe = eda_feats(x); win_t.append(time.perf_counter() - t0)
    X.append(list(fe.values())); y.append(int("-PA4-" in f)); g.append(s)
X, y, g = np.array(X), np.array(y), np.array(g)
tr_t, te_t = [], []
for s in subj:
    tr, te = g != s, g == s
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    t0 = time.perf_counter(); m.fit(X[tr], y[tr]); tr_t.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    for i in np.where(te)[0]: m.predict(X[i:i + 1])
    te_t.append((time.perf_counter() - t0) / te.sum())
res = {"participants": len(subj), "windows": int(len(y)),
       "feature_ms_per_window": round(1000 * float(np.mean(win_t)), 4),
       "csv_read_ms_per_window": round(1000 * float(np.mean(read_t)), 3),
       "train_s_per_fold": round(float(np.mean(tr_t)), 4),
       "test_ms_per_window": round(1000 * float(np.mean(te_t)), 4),
       "total_ms_per_window_excl_io": round(1000 * float(np.mean(win_t) + np.mean(te_t)), 4),
       "parameters": int(X.shape[1] + 1 + 2 * X.shape[1])}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2); print(json.dumps(res, indent=2))
