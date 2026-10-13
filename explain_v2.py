"""
Explainability figures from a finished run (stgraph_v2.py or finetune_v2.py).

  fig_spatial_overlay.png   mean spatial attention on the mean aligned face:
                            BL1, PA4, and PA4 minus BL1 (correctly classified trials)
  fig_temporal.png          mean temporal attention over the 5.5 s clip, BL1 vs PA4
  fig_examples.png          individual PA4 trials: frames with the trial's own
                            attention map (opacity = that frame's temporal weight)
  fig_faithfulness.png      accuracy after blanking the most-attended vs random cells
  fig_subject_acc.png       per-participant accuracy
  explain_summary.json      the numbers behind the figures

Run:
  python explain_v2.py --run_dir runs/graph --crop_dir /scratch/big/home/leibag/crops_v2
"""
import os, glob, json, argparse
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--crop_dir", required=True)
p.add_argument("--out_dir", default=None)
p.add_argument("--fps", type=float, default=25.0)
a = p.parse_args()
out = a.out_dir or os.path.join(a.run_dir, "figs")
os.makedirs(out, exist_ok=True)

Z = [np.load(f) for f in sorted(glob.glob(os.path.join(a.run_dir, "fold_*.npz")))]
keys = np.concatenate([z["keys"] for z in Z]); y = np.concatenate([z["y"] for z in Z])
prob = np.concatenate([z["prob"] for z in Z])
A_s = np.concatenate([z["a_s"] for z in Z]); A_t = np.concatenate([z["a_t"] for z in Z])
T = A_t.shape[1]
correct = (prob > 0.5).astype(int) == y
print(f"trials {len(y)}  accuracy {correct.mean():.4f}")

load = lambda k: np.load(os.path.join(a.crop_dir, k + ".npz"))
z0 = load(keys[0]); S = z0["crops"].shape[1]
vid_idx = z0["idx"]                                   # sampled video frames (28)
sel = np.linspace(0, len(vid_idx) - 1, T).round().astype(int)
t_sec = vid_idx[sel] / a.fps                           # time of each attended frame
mid = z0["crops"].shape[0] // 2
rng = np.random.default_rng(0)
face = np.mean([load(k)["crops"][mid].astype(np.float32) for k in rng.choice(keys, min(300, len(keys)), replace=False)], 0) / 255
up = lambda m: cv2.resize(m.reshape(7, 7).astype(np.float32), (S, S), interpolation=cv2.INTER_CUBIC)

# ---- 1. spatial overlay ----
mb, mp = A_s[(y == 0) & correct].mean(0), A_s[(y == 1) & correct].mean(0)
fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
vmax = max(mb.max(), mp.max())
for i, (m, t) in enumerate([(mb, "BL1 (no pain)"), (mp, "PA4 (high pain)")]):
    ax[i].imshow(face); h = ax[i].imshow(up(m), cmap="jet", alpha=0.45, vmin=0, vmax=vmax)
    ax[i].set_title(f"{t}\nmean spatial attention"); ax[i].axis("off")
plt.colorbar(h, ax=ax[:2], fraction=0.03, label="attention (uniform = 1/49 = 0.020)")
d = mp - mb; lim = np.abs(d).max()
ax[2].imshow(face); h2 = ax[2].imshow(up(d), cmap="RdBu_r", alpha=0.55, vmin=-lim, vmax=lim)
ax[2].set_title("PA4 minus BL1"); ax[2].axis("off")
plt.colorbar(h2, ax=ax[2], fraction=0.046)
plt.savefig(os.path.join(out, "fig_spatial_overlay.png"), dpi=200, bbox_inches="tight"); plt.close()

# ---- 2. temporal attention ----
fig, ax = plt.subplots(figsize=(7, 3.6))
for c, lab, col in [(0, "BL1", "tab:blue"), (1, "PA4", "tab:red")]:
    m = (y == c) & correct
    mu, se = A_t[m].mean(0), A_t[m].std(0) / np.sqrt(m.sum())
    ax.plot(t_sec, mu, color=col, label=f"{lab} (n={m.sum()})"); ax.fill_between(t_sec, mu - se, mu + se, color=col, alpha=0.25)
ax.axhline(1 / T, color="gray", ls="--", lw=1, label="uniform")
ax.set_xlabel("time in clip (s)"); ax.set_ylabel("temporal attention"); ax.legend(frameon=False)
ax.set_title("Which moments of the clip the model attends to")
plt.savefig(os.path.join(out, "fig_temporal.png"), dpi=200, bbox_inches="tight"); plt.close()

# ---- 3. individual examples (most confident correct PA4 trials) ----
ex = np.where((y == 1) & correct)[0]
ex = ex[np.argsort(-prob[ex])][:3]
nf = 6
fig, ax = plt.subplots(len(ex), nf + 1, figsize=(2.1 * (nf + 1), 2.3 * len(ex)), gridspec_kw={"width_ratios": [1] * nf + [1.6]})
ax = np.atleast_2d(ax)
for r, i in enumerate(ex):
    crops = load(keys[i])["crops"]
    cols = np.linspace(0, T - 1, nf).round().astype(int)
    heat = up(A_s[i])
    for c, t in enumerate(cols):
        ax[r, c].imshow(crops[sel[t]])
        ax[r, c].imshow(heat, cmap="jet", alpha=0.15 + 0.55 * A_t[i, t] / A_t[i].max())
        ax[r, c].set_title(f"{t_sec[t]:.1f} s", fontsize=8); ax[r, c].axis("off")
    ax[r, nf].bar(t_sec, A_t[i], width=t_sec[1] - t_sec[0], color="tab:red")
    ax[r, nf].set_title(f"{keys[i]}\np(PA4)={prob[i]:.2f}", fontsize=7); ax[r, nf].tick_params(labelsize=7)
plt.tight_layout(); plt.savefig(os.path.join(out, "fig_examples.png"), dpi=200); plt.close()

# ---- 4. faithfulness ----
summ = {"trials": int(len(y)), "pooled_acc": float(correct.mean())}
log = os.path.join(a.run_dir, "folds.jsonl")
R = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
if R and "acc_top5" in R[0]:
    names = ["acc", "acc_top5", "acc_rand5", "acc_top10", "acc_rand10"]
    lab = ["full input", "top-5 cells\nremoved", "random 5\nremoved", "top-10 cells\nremoved", "random 10\nremoved"]
    vals = np.array([[r[n] for n in names] for r in R])
    mu, se = vals.mean(0), vals.std(0) / np.sqrt(len(vals))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(lab, mu, yerr=se, color=["gray", "tab:red", "tab:blue", "tab:red", "tab:blue"], capsize=3)
    ax.axhline(0.5, color="k", ls="--", lw=1); ax.set_ylim(0.4, max(0.75, mu.max() + 0.05))
    ax.set_ylabel("accuracy (mean over folds)"); ax.set_title("Faithfulness: removing attended vs random face cells")
    plt.savefig(os.path.join(out, "fig_faithfulness.png"), dpi=200, bbox_inches="tight"); plt.close()
    summ["faithfulness"] = dict(zip(names, mu.round(4).tolist()))

# ---- 5. per-participant accuracy ----
subj = np.array([k.split("-")[0] for k in keys])
sa = np.array([correct[subj == s].mean() for s in np.unique(subj)])
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.hist(sa, bins=np.arange(0, 1.05, 0.05), color="tab:blue", edgecolor="white")
ax.axvline(0.5, color="k", ls="--", lw=1, label="chance"); ax.axvline(sa.mean(), color="tab:red", lw=1.5, label=f"mean {sa.mean():.3f}")
ax.set_xlabel("accuracy of held-out participant"); ax.set_ylabel("participants"); ax.legend(frameon=False)
plt.savefig(os.path.join(out, "fig_subject_acc.png"), dpi=200, bbox_inches="tight"); plt.close()

top = np.argsort(-mp)[:5]
summ.update({"subject_acc_mean": float(sa.mean()), "subject_acc_std": float(sa.std()),
             "pa4_top5_cells_(row,col)_from_top_left": [[int(c // 7) + 1, int(c % 7) + 1] for c in top],
             "pa4_temporal_peak_sec": float(t_sec[np.argmax(A_t[(y == 1) & correct].mean(0))])})
json.dump(summ, open(os.path.join(out, "explain_summary.json"), "w"), indent=2)
print(json.dumps(summ, indent=2))
