"""
Stage 2: graph over spatial cells of the aligned face, per frame.
Nodes = (7x7 grid) x (T frames). Attention over nodes -> saliency map.
"""
import argparse, glob, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

CLASSES = ["BL1", "PA1", "PA2", "PA3", "PA4"]
C2I = {c: i for i, c in enumerate(CLASSES)}
G = 7                       # spatial grid
CELLS = G * G               # 49 cells


def build_index(cache_dir):
    items = []
    for p in sorted(glob.glob(f"{cache_dir}/*_map.npy")):
        base = os.path.basename(p)[:-8]        # strip '_map.npy'
        parts = base.split("-")
        if len(parts) < 3 or parts[1] not in C2I:
            continue
        items.append((p, C2I[parts[1]], parts[0]))
    return items


def load_trial(path, T=28):
    a = np.load(path).astype(np.float32)       # (T,512,7,7)
    n = a.shape[0]
    if n >= T:
        a = a[np.linspace(0, n-1, T).astype(int)]
    else:
        a = np.concatenate([a, np.repeat(a[-1:], T-n, 0)], 0)
    # (T,512,7,7) -> (T,49,512)  nodes = time-major, cell-minor
    a = a.transpose(0, 2, 3, 1).reshape(T, CELLS, 512)
    return torch.from_numpy(a)


def make_adj(T=28, win=2):
    """Windowed temporal adjacency within each cell."""
    N = T * CELLS
    A = torch.zeros(N, N)
    for t in range(T):
        for c in range(CELLS):
            i = t * CELLS + c
            for d in range(-win, win+1):
                if d and 0 <= t+d < T:
                    A[i, (t+d)*CELLS + c] = 1
    A = A + torch.eye(N)
    return A / A.sum(1, keepdim=True)


class SpatialPainNet(nn.Module):
    def __init__(self, T=28, feat=512, hidden=128, nclass=5, win=2):
        super().__init__()
        self.T = T
        self.register_buffer("A", make_adj(T, win))
        self.W1 = nn.Linear(feat, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.attn = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                  nn.Linear(64, 1))
        self.cls = nn.Linear(hidden, nclass)

    def forward(self, x):
        """x: (B,T,49,512) -> logits, per-cell saliency (B,49)"""
        B = x.shape[0]
        h = x.reshape(B, self.T * CELLS, -1)
        h = F.relu(self.W1(torch.einsum("ij,bjc->bic", self.A, h)))
        h = F.relu(self.W2(torch.einsum("ij,bjc->bic", self.A, h)))
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1)     # (B, T*49)
        w_c = w.reshape(B, self.T, CELLS).sum(1)               # (B, 49)
        pooled = torch.einsum("bn,bnh->bh", w, h)
        return self.cls(pooled), w_c


def fit(model, idx, device, epochs, bs, lr=1e-3, T=28):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(len(idx))
        tot = corr = n = 0
        for i in range(0, len(order), bs):
            b = [idx[j] for j in order[i:i+bs]]
            xs = torch.stack([load_trial(p, T) for p, _, _ in b]).to(device)
            ys = torch.tensor([y for _, y, _ in b]).to(device)
            opt.zero_grad()
            logits, _ = model(xs)
            loss = crit(logits, ys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()*len(b); corr += (logits.argmax(1)==ys).sum().item()
            n += len(b)
        sched.step(tot/n)
        print(f"    ep {ep+1:02d}/{epochs}  loss {tot/n:.4f}  train acc {corr/n:.3f}")


@torch.no_grad()
def predict(model, idx, device, bs=16, T=28):
    model.eval()
    P, Tr, W = [], [], []
    for i in range(0, len(idx), bs):
        b = idx[i:i+bs]
        xs = torch.stack([load_trial(p, T) for p, _, _ in b]).to(device)
        logits, w_c = model(xs)
        P.extend(logits.argmax(1).cpu().tolist())
        Tr.extend([y for _, y, _ in b])
        W.append(w_c.cpu().numpy())
    return np.array(Tr), np.array(P), np.concatenate(W, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/biovid/feats_spatial"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--frames", type=int, default=28)
    ap.add_argument("--max_folds", type=int, default=0)
    ap.add_argument("--binary", action="store_true",
                    help="keep only BL1 vs PA4 (2 classes)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.expanduser("~/biovid/st_results"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    idx = build_index(args.cache)
    nclass = 5
    class_names = CLASSES
    if args.binary:
        keep = {0, 4}                      # BL1=0, PA4=4
        idx = [(p, 0 if y == 0 else 1, s) for (p, y, s) in idx if y in keep]
        nclass, class_names = 2, ["BL1", "PA4"]
        print("binary mode: BL1 vs PA4")
    print(f"trials indexed: {len(idx)} | device {device} | nclass {nclass}")

    by_sub = {}
    for it in idx:
        by_sub.setdefault(it[2], []).append(it)
    subs = sorted(by_sub)
    if args.max_folds:
        subs = subs[:args.max_folds]
    print(f"subjects: {len(subs)} -> {[len(by_sub[s]) for s in subs]}")

    AT, AP, AW, accs = [], [], [], []
    for k, held in enumerate(subs):
        tr = [it for s in subs if s != held for it in by_sub[s]]
        te = by_sub[held]
        m = SpatialPainNet(T=args.frames, nclass=nclass).to(device)
        fit(m, tr, device, args.epochs, args.batch, args.lr, args.frames)
        t, p, w = predict(m, te, device, T=args.frames)
        AT.extend(t.tolist()); AP.extend(p.tolist()); AW.append(w)
        acc = accuracy_score(t, p); accs.append(acc)
        print(f"  [{k+1}/{len(subs)}] {held}: n={len(te)} acc={acc:.3f}")
        with open(f"{args.out}/folds.csv", "a") as fh:
            fh.write(f"{held},{acc:.4f}\n")

    AT, AP = np.array(AT), np.array(AP)
    Wm = np.concatenate(AW, 0).mean(0).reshape(G, G)     # 7x7 saliency map
    mac = precision_recall_fscore_support(AT, AP, average="macro", zero_division=0)
    per = precision_recall_fscore_support(AT, AP, average=None, zero_division=0)
    cm = confusion_matrix(AT, AP, labels=list(range(nclass)))

    print("\n" + "="*60)
    print(f"LOSO {len(subs)} folds  acc {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"macro P/R/F1 {mac[0]:.4f} / {mac[1]:.4f} / {mac[2]:.4f}")
    print("\nSALIENCY MAP (7x7, mean attention; rows = top->bottom of face):")
    for r in range(G):
        print("  " + "  ".join(f"{Wm[r,c]:.3f}" for c in range(G)))
    print("\nper-class:")
    for c, nm in enumerate(class_names):
        print(f"  {nm}: P={per[0][c]:.3f} R={per[1][c]:.3f} F1={per[2][c]:.3f}")
    print("\nconfusion (rows=true):")
    print(cm)

    res = {"n_folds": len(subs), "n_trials": int(len(AT)),
           "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
           "fold_acc": [float(a) for a in accs],
           "macro_f1": float(mac[2]),
           "nclass": nclass,
           "per_class": {n: {"precision": float(per[0][i]),
                             "recall": float(per[1][i]),
                             "f1": float(per[2][i])}
                         for i, n in enumerate(class_names)},
           "saliency_map": Wm.tolist(),
           "confusion_matrix": cm.tolist()}
    with open(f"{args.out}/st_spatial.json", "w") as f:
        json.dump(res, f, indent=2)
    np.save(f"{args.out}/saliency_map.npy", Wm)
    print(f"\nsaved -> {args.out}/st_spatial.json")


if __name__ == "__main__":
    main()
