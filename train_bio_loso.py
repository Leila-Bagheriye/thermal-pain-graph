import os, glob, json, argparse, time
import numpy as np, pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

p = argparse.ArgumentParser()
p.add_argument("--root", default=os.path.expanduser("~/biovid_bio"))
p.add_argument("--classes", default="BL1,PA4")
p.add_argument("--mods", default="gsr,ecg")          # try: gsr | ecg | gsr,ecg | gsr,ecg,emg
p.add_argument("--model", default="rf")              # rf | lr
p.add_argument("--subjnorm", type=int, default=1)    # per-subject z-scoring of features (no labels used)
p.add_argument("--out", default="bio_loso_results.jsonl")
a = p.parse_args()
FS = 512
CLS = a.classes.split(",")
MODS = a.mods.split(",")

# ---------- feature extraction ----------
def eda_feats(x):
    t = np.arange(len(x)) / FS
    d = np.diff(x) * FS
    base = x[:FS].mean()                      # first second as baseline
    return {"eda_mean": x.mean(), "eda_std": x.std(), "eda_max": x.max(), "eda_min": x.min(),
            "eda_range": x.max() - x.min(), "eda_rise": x.max() - base,
            "eda_end_minus_start": x[-FS:].mean() - base,
            "eda_slope": linregress(t, x).slope, "eda_argmax_t": np.argmax(x) / FS,
            "eda_auc": np.trapz(x - base, t), "eda_d_mean": d.mean(), "eda_d_max": d.max(),
            "eda_d_std": d.std()}

def ecg_feats(x):
    b, a_ = butter(3, [5 / (FS / 2), 30 / (FS / 2)], btype="band")
    y = filtfilt(b, a_, x)
    if abs(y.min()) > abs(y.max()): y = -y          # handle inverted polarity
    pk, _ = find_peaks(y, distance=int(0.33 * FS), height=np.percentile(y, 90) * 0.5)
    f = {k: np.nan for k in ["hr_mean", "hr_max", "hr_min", "hr_std", "sdnn", "rmssd", "hr_slope", "n_beats"]}
    f["n_beats"] = len(pk)
    if len(pk) >= 4:
        rr = np.diff(pk) / FS
        rr = rr[(rr > 0.3) & (rr < 2.0)]
        if len(rr) >= 3:
            hr = 60 / rr
            f.update(hr_mean=hr.mean(), hr_max=hr.max(), hr_min=hr.min(), hr_std=hr.std(),
                     sdnn=rr.std() * 1000, rmssd=np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000,
                     hr_slope=linregress(np.arange(len(hr)), hr).slope)
    return f

def emg_feats(x):
    r = np.abs(x - x.mean())
    return {"emg_rms": np.sqrt(np.mean(x ** 2)), "emg_mav": r.mean(), "emg_std": x.std(),
            "emg_max": r.max(), "emg_late_minus_early": r[-2 * FS:].mean() - r[:FS].mean()}

def read_sig(f):
    df = pd.read_csv(f, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    return df

cache = os.path.join(a.root, f"feats_{'_'.join(CLS)}.csv")
if os.path.exists(cache):
    F = pd.read_csv(cache)
else:
    rows = []
    files = glob.glob(os.path.join(a.root, "**", "*.csv"), recursive=True)
    for i, f in enumerate(files):
        base = os.path.basename(f)
        lab = next((c for c in CLS if f"-{c}-" in base), None)
        if lab is None: continue
        df = read_sig(f)
        r = {"subject": base.split("-")[0], "label": CLS.index(lab), "file": base}
        if "gsr" in df: r.update(eda_feats(df["gsr"].values.astype(float)))
        if "ecg" in df: r.update(ecg_feats(df["ecg"].values.astype(float)))
        emgc = [c for c in df.columns if c.startswith("emg")]
        if emgc: r.update(emg_feats(df[emgc[0]].values.astype(float)))
        rows.append(r)
        if i % 500 == 0: print(f"features {i}/{len(files)}", flush=True)
    F = pd.DataFrame(rows); F.to_csv(cache, index=False)
print("Samples:", len(F), "Subjects:", F.subject.nunique(), "Class counts:", F.label.value_counts().to_dict())

prefix = {"gsr": "eda_", "ecg": ("hr_", "sdnn", "rmssd", "n_beats"), "emg": "emg_"}
cols = [c for c in F.columns if any(c.startswith(prefix[m]) if isinstance(prefix[m], str)
                                     else c.startswith(prefix[m]) for m in MODS)]
X = F[cols].copy()
X = X.fillna(X.median())
if a.subjnorm:   # standardise each feature within each participant (unsupervised, removes individual baselines)
    X = X.groupby(F.subject).transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
X = X.values; y = F.label.values; g = F.subject.values
print("Features used:", cols)

# ---------- LOSO, crash-safe ----------
tag = f"{a.mods}|{a.model}|norm{a.subjnorm}|{a.classes}"
done = set()
if os.path.exists(a.out):
    for line in open(a.out):
        r = json.loads(line)
        if r["tag"] == tag: done.add(r["subject"])

def make():
    if a.model == "lr": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=3, n_jobs=-1, random_state=0)

for s in sorted(set(g)):
    if s in done: continue
    tr, te = g != s, g == s
    m = make().fit(X[tr], y[tr])
    pr = m.predict(X[te])
    res = {"tag": tag, "subject": s, "acc": accuracy_score(y[te], pr),
           "f1": f1_score(y[te], pr, average="macro"), "n": int(te.sum()),
           "y": y[te].tolist(), "pred": pr.tolist()}
    with open(a.out, "a") as fo: fo.write(json.dumps(res) + "\n")
    print(f"{s}: acc={res['acc']:.3f}", flush=True)

R = [json.loads(l) for l in open(a.out) if json.loads(l)["tag"] == tag]
acc = np.array([r["acc"] for r in R]); f1 = np.array([r["f1"] for r in R])
print(f"\n[{tag}] folds={len(R)}  LOSO acc = {acc.mean():.4f} ± {acc.std():.4f}   macro-F1 = {f1.mean():.4f}")
if a.model == "rf":
    imp = make().fit(X, y).feature_importances_
    print("Top features:", sorted(zip(cols, imp), key=lambda t: -t[1])[:10])
