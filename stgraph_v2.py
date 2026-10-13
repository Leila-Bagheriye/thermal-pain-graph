"""
Stage 2 - frozen ResNet-18 features + spatio-temporal graph, leave-one-subject-out.

Backbone is EXPLICIT: torchvision ResNet-18, ImageNet weights (IMAGENET1K_V1),
truncated before global pooling -> (512, 7, 7) per frame. Features are computed
once in memory at start-up (no large files written).

Models (--model):
  graph    full STGraph (spatial + temporal edges, factorised attention)
  nograph  same network without graph layers (ablation)
  linear   mean-pooled features -> PCA(50) -> logistic regression (baseline)

Sanity check: --shuffle_labels 1 trains on shuffled labels; accuracy must be ~0.50.
If it is clearly above 0.50, something leaks.

Outputs in --out_dir:
  config.json, folds.jsonl (one line per held-out participant),
  fold_<subject>.npz (per-trial prob, label, spatial + temporal attention),
  summary.json. Restartable: finished folds are skipped.

Faithfulness test (graph / nograph): at test time the 5 or 10 cells with the
highest attention are blanked; the accuracy drop is compared with blanking
the same number of random cells. A faithful explanation gives a larger drop.
"""
import os, glob, json, argparse, time
import numpy as np, torch, torch.nn.functional as F, torchvision
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from models_v2 import STGraph

p = argparse.ArgumentParser()
p.add_argument("--crop_dir", required=True)
p.add_argument("--out_dir", default="runs/graph")
p.add_argument("--model", default="graph", choices=["graph", "nograph", "linear"])
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--win", type=int, default=2)
p.add_argument("--subjnorm", type=int, default=1)
p.add_argument("--shuffle_labels", type=int, default=0)
p.add_argument("--max_folds", type=int, default=0, help="0 = all participants")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--backbone", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
p.add_argument("--n_frames", type=int, default=0, help="0 = all 28 stored frames; else evenly subsample")
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = vars(a).copy()
cfg_path = os.path.join(a.out_dir, "config.json")
if os.path.exists(cfg_path) and json.load(open(cfg_path)) != cfg:
    raise SystemExit(f"{a.out_dir} holds a run with different settings - use a new --out_dir")
json.dump(cfg, open(cfg_path, "w"), indent=2)

# ---------------- data + frozen features ----------------
files = sorted(glob.glob(os.path.join(a.crop_dir, "*.npz")))
keys = [os.path.basename(f)[:-4] for f in files]
y = np.array([0 if "-BL1-" in k else 1 for k in keys])
subj = np.array([k.split("-")[0] for k in keys])
print(f"trials {len(keys)}  subjects {len(set(subj))}  BL1 {int((y==0).sum())}  PA4 {int((y==1).sum())}", flush=True)

if a.backbone == "resnet18":
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    trunk = torch.nn.Sequential(*list(torchvision.models.resnet18(weights=w).children())[:-2])
else:  # MobileNetV3-Small, ImageNet weights: (576, 7, 7) at 224 px
    w = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    trunk = torchvision.models.mobilenet_v3_small(weights=w).features
trunk = trunk.eval().to(dev)
print("backbone:", a.backbone, "params:", sum(q.numel() for q in trunk.parameters()), flush=True)
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
X = None
t0 = time.time()
with torch.no_grad():
    for i, f in enumerate(files):
        c = torch.from_numpy(np.load(f)["crops"]).to(dev).permute(0, 3, 1, 2).float() / 255
        c = (F.interpolate(c, size=224, mode="bilinear", align_corners=False) - mean) / std
        fm = trunk(c)                                        # (T, 512, 7, 7)
        fm = fm.flatten(2).transpose(1, 2).half().cpu().numpy()  # (T, 49, 512)
        if X is None:
            X = np.zeros((len(files),) + fm.shape, np.float16)
            print("feature map per trial:", fm.shape, "(frames, cells, channels)", flush=True)
        X[i] = fm
        if i % 500 == 0:
            print(f"features {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
if a.n_frames:
    sel = np.linspace(0, X.shape[1] - 1, a.n_frames).round().astype(int)
    X = np.ascontiguousarray(X[:, sel])
    print("frames used:", sel.tolist(), flush=True)
T = X.shape[1]
del trunk
torch.cuda.empty_cache()

if a.subjnorm:  # per participant, per channel; uses no labels
    for s in np.unique(subj):
        m = subj == s
        xs = X[m].astype(np.float32)
        mu, sd = xs.mean((0, 1, 2)), xs.std((0, 1, 2)) + 1e-6
        X[m] = ((xs - mu) / sd).astype(np.float16)


# ---------------- one fold ----------------
def evaluate(model, idx, mask=None):
    probs, A_s, A_t = [], [], []
    model.eval()
    with torch.no_grad():
        for b in range(0, len(idx), 64):
            xb = torch.from_numpy(X[idx[b:b + 64]]).to(dev).float()
            if mask is not None:
                xb = xb * (1 - mask[b:b + 64])[:, None, :, None]
            lo, s_, t_ = model(xb)
            probs.append(torch.softmax(lo, -1)[:, 1].cpu()); A_s.append(s_.cpu()); A_t.append(t_.cpu())
    return torch.cat(probs).numpy(), torch.cat(A_s), torch.cat(A_t)


def run_fold(tr, te, rng):
    ytr = y[tr].copy()
    if a.shuffle_labels:
        ytr = rng.permutation(ytr)
    if a.model == "linear":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        pooled = lambda ii: X[ii].astype(np.float32).mean((1, 2))
        m = make_pipeline(StandardScaler(), PCA(50, random_state=0), LogisticRegression(max_iter=3000, C=0.1))
        m.fit(pooled(tr), ytr)
        return {"prob": m.predict_proba(pooled(te))[:, 1]}

    torch.manual_seed(a.seed)
    model = STGraph(C=X.shape[3], T=T, H=a.hidden, win=a.win, use_graph=(a.model == "graph")).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(len(tr))
        tot = 0.0
        for b in range(0, len(perm), a.bs):
            bi = perm[b:b + a.bs]
            xb = torch.from_numpy(X[tr[bi]]).to(dev).float()
            if rng.random() < 0.5:  # horizontal flip of the 7x7 grid
                B_ = xb.shape[0]
                xb = xb.view(B_, T, 7, 7, -1).flip(3).reshape(B_, T, 49, -1)
            yb = torch.from_numpy(ytr[bi]).to(dev)
            loss = F.cross_entropy(model(xb)[0], yb, label_smoothing=0.1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item() * len(bi)
        sched.step()
    # fixed number of epochs: the test participant is never used for model selection
    prob, A_s, A_t = evaluate(model, te)
    out = {"prob": prob, "a_s": A_s.numpy(), "a_t": A_t.numpy(), "train_loss": tot / len(tr)}
    for k in (5, 10):  # faithfulness: blank the k most-attended cells vs k random cells
        top = torch.zeros_like(A_s).scatter_(1, A_s.topk(k, 1).indices, 1.0).to(dev)
        out[f"acc_top{k}"] = accuracy_score(y[te], evaluate(model, te, top)[0] > 0.5)
        accs = []
        for _ in range(5):
            r = torch.zeros_like(A_s)
            for i in range(len(r)):
                r[i, rng.choice(49, k, replace=False)] = 1
            accs.append(accuracy_score(y[te], evaluate(model, te, r.to(dev))[0] > 0.5))
        out[f"acc_rand{k}"] = float(np.mean(accs))
    return out


# ---------------- leave-one-subject-out ----------------
subjects = sorted(set(subj))
if a.max_folds:
    subjects = subjects[:a.max_folds]
log = os.path.join(a.out_dir, "folds.jsonl")
for s in subjects:
    fz = os.path.join(a.out_dir, f"fold_{s}.npz")
    if os.path.exists(fz):
        continue
    t1 = time.time()
    tr, te = np.where(subj != s)[0], np.where(subj == s)[0]
    o = run_fold(tr, te, np.random.default_rng(a.seed + len(s)))
    pred = (o["prob"] > 0.5).astype(int)
    rec = {"subject": s, "n": int(len(te)), "acc": accuracy_score(y[te], pred),
           "f1": f1_score(y[te], pred, average="macro"), "sec": round(time.time() - t1)}
    rec.update({k: float(v) for k, v in o.items() if k.startswith("acc_") or k == "train_loss"})
    np.savez(fz, keys=np.array(keys)[te], y=y[te], **{k: v for k, v in o.items() if isinstance(v, np.ndarray)})
    open(log, "a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)

# ---------------- summary ----------------
R = [json.loads(l) for l in open(log)]
Y, P = [], []
for s in subjects:
    z = np.load(os.path.join(a.out_dir, f"fold_{s}.npz"))
    Y.append(z["y"]); P.append(z["prob"] > 0.5)
Y, P = np.concatenate(Y), np.concatenate(P).astype(int)
acc = np.array([r["acc"] for r in R])
summ = {"model": a.model, "folds": len(R), "acc_mean": acc.mean(), "acc_std": acc.std(),
        "pooled_acc": accuracy_score(Y, P), "pooled_macro_f1": f1_score(Y, P, average="macro"),
        "confusion_[[TN,FP],[FN,TP]]": confusion_matrix(Y, P).tolist()}
for k in ("acc_top5", "acc_rand5", "acc_top10", "acc_rand10"):
    if k in R[0]:
        summ[k] = float(np.mean([r[k] for r in R]))
json.dump(summ, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2, default=float)
print(json.dumps(summ, indent=2, default=float))
