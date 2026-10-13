"""
Stage 2b - end-to-end fine-tuning: ResNet-18 (last block trainable) + STGraph head.

Fine-tuning a CNN inside all 87 LOSO folds is too expensive, so this uses
subject-grouped 5-fold cross-validation: every participant is tested exactly once,
and no participant is ever in both training and test of the same fold.
Report it as "5-fold subject-independent CV", not as LOSO.

Uses n_frames (default 8) evenly spaced frames per trial to keep memory low.
Outputs per fold: fold_<k>.npz (keys, y, prob, a_s, a_t) and folds.jsonl, summary.json.
"""
import os, glob, json, argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from models_v2 import STGraph

p = argparse.ArgumentParser()
p.add_argument("--crop_dir", required=True)
p.add_argument("--out_dir", default="runs/finetune")
p.add_argument("--n_frames", type=int, default=8)
p.add_argument("--folds", type=int, default=5)
p.add_argument("--epochs", type=int, default=12)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr_backbone", type=float, default=1e-4)
p.add_argument("--lr_head", type=float, default=3e-4)
p.add_argument("--unfreeze", default="layer4", help="comma list, e.g. layer3,layer4")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--loso", type=int, default=0, help="1 = leave-one-subject-out instead of k-fold")
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
json.dump(vars(a), open(os.path.join(a.out_dir, "config.json"), "w"), indent=2)
dev = "cuda"
torch.manual_seed(a.seed)

files = sorted(glob.glob(os.path.join(a.crop_dir, "*.npz")))
keys = np.array([os.path.basename(f)[:-4] for f in files])
y = np.array([0 if "-BL1-" in k else 1 for k in keys])
subj = np.array([k.split("-")[0] for k in keys])
fidx = None
crops = []
for i, f in enumerate(files):
    c = np.load(f)["crops"]
    if fidx is None:
        fidx = np.linspace(0, len(c) - 1, a.n_frames).round().astype(int)
    crops.append(c[fidx])
    if i % 500 == 0:
        print(f"loading {i}/{len(files)}", flush=True)
crops = np.stack(crops)  # (N, F, S, S, 3) uint8
print("crops", crops.shape, flush=True)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.trunk = nn.Sequential(*list(r.children())[:-2])
        names = dict(zip(["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"], self.trunk))
        for prm in self.trunk.parameters():
            prm.requires_grad = False
        for n in a.unfreeze.split(","):
            for prm in names[n].parameters():
                prm.requires_grad = True
        self.head = STGraph(C=512, T=a.n_frames, H=128, win=1)

    def forward(self, x):  # x uint8 (B, F, S, S, 3)
        B, Fr = x.shape[:2]
        x = x.flatten(0, 1).permute(0, 3, 1, 2).float() / 255
        x = (F.interpolate(x, size=224, mode="bilinear", align_corners=False) - MEAN) / STD
        fm = self.trunk(x).flatten(2).transpose(1, 2)            # (B*F, 49, 512)
        return self.head(fm.reshape(B, Fr, 49, -1))


def keep_bn_frozen(m):
    for mod in m.trunk.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()


log = os.path.join(a.out_dir, "folds.jsonl")
rng = np.random.default_rng(a.seed)
if a.loso:
    splits = [(np.where(subj != s)[0], np.where(subj == s)[0]) for s in sorted(set(subj))]
else:
    splits = list(GroupKFold(a.folds).split(keys, y, subj))
print("folds:", len(splits), flush=True)
for k, (tr, te) in enumerate(splits):
    fz = os.path.join(a.out_dir, f"fold_{k}.npz")
    if os.path.exists(fz):
        continue
    t0 = time.time()
    net = Net().to(dev)
    opt = torch.optim.AdamW([
        {"params": [q for q in net.trunk.parameters() if q.requires_grad], "lr": a.lr_backbone},
        {"params": net.head.parameters(), "lr": a.lr_head}], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(a.epochs):
        net.train(); keep_bn_frozen(net)
        perm, tot = rng.permutation(tr), 0.0
        for b in range(0, len(perm), a.bs):
            bi = perm[b:b + a.bs]
            xb = torch.from_numpy(crops[bi]).to(dev)
            if rng.random() < 0.5:
                xb = xb.flip(3)                                   # horizontal flip
            yb = torch.from_numpy(y[bi]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(net(xb)[0], yb, label_smoothing=0.1)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(bi)
        sched.step()
        print(f"fold {k} ep {ep+1}/{a.epochs} train_loss {tot/len(tr):.4f}", flush=True)
    net.eval()
    P, AS, AT = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for b in range(0, len(te), 16):
            lo, s_, t_ = net(torch.from_numpy(crops[te[b:b + 16]]).to(dev))
            P.append(torch.softmax(lo.float(), -1)[:, 1].cpu()); AS.append(s_.float().cpu()); AT.append(t_.float().cpu())
    prob = torch.cat(P).numpy()
    pred = (prob > 0.5).astype(int)
    per_subj = {s: accuracy_score(y[te][subj[te] == s], pred[subj[te] == s]) for s in np.unique(subj[te])}
    rec = {"fold": k, "n_test": int(len(te)), "test_subjects": len(per_subj),
           "acc": accuracy_score(y[te], pred), "f1": f1_score(y[te], pred, average="macro"),
           "subject_acc_mean": float(np.mean(list(per_subj.values()))), "sec": round(time.time() - t0)}
    np.savez(fz, keys=keys[te], y=y[te], prob=prob, a_s=torch.cat(AS).numpy(), a_t=torch.cat(AT).numpy())
    open(log, "a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)

R = [json.loads(l) for l in open(log)]
summ = {"folds": len(R), "acc_mean": float(np.mean([r["acc"] for r in R])),
        "acc_std": float(np.std([r["acc"] for r in R])),
        "subject_acc_mean": float(np.mean([r["subject_acc_mean"] for r in R]))}
json.dump(summ, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)
print(json.dumps(summ, indent=2))
