import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

F = pd.read_csv("feats_BL1_PA4.csv")
cols = [c for c in F.columns if c.startswith("eda_")]
X = F[cols].fillna(F[cols].median())
X = X.groupby(F.subject).transform(lambda s: (s - s.mean()) / (s.std() + 1e-8)).values
y, g = F.label.values, F.subject.values
subs = np.array(sorted(set(g)))

for name, make in [("LR", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))),
                   ("RF", lambda: RandomForestClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=0))]:
    accs, f1s = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        te_s = rng.choice(subs, 17, replace=False)
        te = np.isin(g, te_s)
        p = make().fit(X[~te], y[~te]).predict(X[te])
        accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, average="macro"))
        print(f"{name} seed {seed}: acc={accs[-1]:.4f}")
    print(f"\n{name} 10x 70/17: acc = {np.mean(accs):.4f} ± {np.std(accs):.4f} "
          f"(min {min(accs):.3f}, max {max(accs):.3f}), F1 = {np.mean(f1s):.4f}\n")
