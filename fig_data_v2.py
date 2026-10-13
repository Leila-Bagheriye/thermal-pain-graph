"""
Data and architecture figures for the paper, built from the real BioVid files.

  fig_pipeline.png      raw frame + MTCNN box/landmarks -> aligned crops over time
  fig_samples.png       BL1 vs PA4 crops for 4 participants + mean faces + difference
  fig_biosignals.png    EDA and ECG, mean +/- SE, BL1 vs PA4
  fig_architecture.png  two-branch architecture with fusion (matches models_v2.py)

Check the BioVid licence (Agreement.pdf) before publishing participant faces;
use --blur 1 to blur individual faces (mean faces are not identifiable).

Run:
  python fig_data_v2.py --video_dir ../biovid/clips --crop_dir ../crops_v2 \
                        --bio_dir ../biovid_bio --out_dir figs_data
"""
import os, glob, argparse
import numpy as np, cv2, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

p = argparse.ArgumentParser()
p.add_argument("--video_dir", required=True)
p.add_argument("--crop_dir", required=True)
p.add_argument("--bio_dir", required=True)
p.add_argument("--out_dir", default="figs_data")
p.add_argument("--blur", type=int, default=0)
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
blur = (lambda im: cv2.GaussianBlur(im, (0, 0), 6)) if a.blur else (lambda im: im)
FPS = 25.0

crops = sorted(glob.glob(os.path.join(a.crop_dir, "*.npz")))
key = lambda f: os.path.basename(f)[:-4]
subj = lambda k: k.split("-")[0]

# ---------- 1. pipeline ----------
k = key(next(f for f in crops if "-PA4-" in f))
vid = glob.glob(os.path.join(a.video_dir, "**", k + ".mp4"), recursive=True)[0]
z = np.load(os.path.join(a.crop_dir, k + ".npz"))
cap = cv2.VideoCapture(vid); cap.set(cv2.CAP_PROP_POS_FRAMES, int(z["idx"][14]))
_, fr = cap.read(); cap.release()
rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
try:
    from facenet_pytorch import MTCNN
    sc = 640 / rgb.shape[1]
    b, _, lm = MTCNN(keep_all=True, device="cpu").detect(cv2.resize(rgb, None, fx=sc, fy=sc), landmarks=True)
    i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))); b, lm = b[i] / sc, lm[i] / sc
except Exception:
    b = lm = None
fig = plt.figure(figsize=(13, 4.2))
ax0 = fig.add_axes([0.0, 0.05, 0.33, 0.85]); ax0.imshow(blur(rgb)); ax0.axis("off")
if b is not None:
    ax0.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="lime", lw=2))
    ax0.scatter(lm[:, 0], lm[:, 1], c="red", s=14)
ax0.set_title("(a) raw frame, MTCNN box + 5 landmarks", fontsize=10)
sel = np.linspace(0, len(z["crops"]) - 1, 7).round().astype(int)
for j, t in enumerate(sel):
    ax = fig.add_axes([0.36 + j * 0.091, 0.25, 0.088, 0.5]); ax.imshow(blur(z["crops"][t])); ax.axis("off")
    ax.set_title(f"{z['idx'][t] / FPS:.1f} s", fontsize=8)
fig.text(0.68, 0.85, "(b) aligned 112 x 112 crops, 28 frames over 5.5 s (7 shown)", ha="center", fontsize=10)
fig.savefig(os.path.join(a.out_dir, "fig_pipeline.png"), dpi=220, bbox_inches="tight"); plt.close()

# ---------- 2. samples + mean faces ----------
S = sorted({subj(key(f)) for f in crops})[:4]
fig, ax = plt.subplots(3, 4, figsize=(9, 7))
for c, s in enumerate(S):
    for r, lab in enumerate(["BL1", "PA4"]):
        f = sorted(glob.glob(os.path.join(a.crop_dir, f"{s}-{lab}-*.npz")))[0]
        ax[r, c].imshow(blur(np.load(f)["crops"][20])); ax[r, c].axis("off")
        ax[r, c].set_title(f"{lab}" + (f"\n{s}" if r == 0 else ""), fontsize=8)
rng = np.random.default_rng(0)
mean = {}
for lab in ["BL1", "PA4"]:
    fs = [f for f in crops if f"-{lab}-" in f]
    fs = rng.choice(fs, min(400, len(fs)), replace=False)
    mean[lab] = np.mean([np.load(f)["crops"][14:28].mean(0) for f in fs], 0) / 255
ax[2, 0].imshow(mean["BL1"]); ax[2, 0].set_title("mean BL1", fontsize=8)
ax[2, 1].imshow(mean["PA4"]); ax[2, 1].set_title("mean PA4", fontsize=8)
d = (mean["PA4"] - mean["BL1"]).mean(-1); lim = np.abs(d).max()
ax[2, 2].imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim); ax[2, 2].set_title("PA4 - BL1 (2nd half of clip)", fontsize=8)
for x in ax[2]: x.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "fig_samples.png"), dpi=220); plt.close()

# ---------- 3. biosignals ----------
bf = glob.glob(os.path.join(a.bio_dir, "**", "*_bio.csv"), recursive=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for lab, col in [("BL1", "tab:blue"), ("PA4", "tab:red")]:
    fs = [f for f in bf if f"-{lab}-" in os.path.basename(f)]
    fs = rng.choice(fs, min(400, len(fs)), replace=False)
    E, H = [], []
    for f in fs:
        df = pd.read_csv(f, sep=None, engine="python"); df.columns = [c.strip().lower() for c in df.columns]
        e = df["gsr"].values.astype(float); E.append(e - e[:512].mean())
        x = df["ecg"].values.astype(float); H.append((x - x.mean()) / (x.std() + 1e-9))
    n = min(map(len, E)); E = np.array([e[:n] for e in E]); t = np.arange(n) / 512
    mu, se = E.mean(0), E.std(0) / np.sqrt(len(E))
    ax[0].plot(t, mu, color=col, label=f"{lab} (n={len(E)})"); ax[0].fill_between(t, mu - se, mu + se, color=col, alpha=0.25)
    ax[1].plot(t[:1024], H[0][:1024], color=col, lw=0.8, label=f"{lab} example")
ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("EDA change from first second (uS)"); ax[0].legend(frameon=False)
ax[0].set_title("Skin conductance, mean +/- SE")
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("ECG (z-scored)"); ax[1].legend(frameon=False); ax[1].set_title("ECG, first 2 s of one trial each")
plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "fig_biosignals.png"), dpi=220); plt.close()

# ---------- 4. architecture ----------
fig, ax = plt.subplots(figsize=(14, 5.6)); ax.set_xlim(0, 14); ax.set_ylim(0, 5.6); ax.axis("off")
def box(x, y, w, h, txt, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5)
def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color="#333"))
V, W, F, X = "#cfe2f3", "#d9ead3", "#fce5cd", "#fff2cc"
ax.text(0.1, 5.3, "Contactless branch (video)", fontsize=10, weight="bold")
box(0.1, 3.6, 1.5, 1.2, "Video\n28 frames\n5.5 s", V); arr(1.6, 4.2, 1.9, 4.2)
box(1.9, 3.6, 1.6, 1.2, "MTCNN\ndetect + align\n112x112", V); arr(3.5, 4.2, 3.8, 4.2)
box(3.8, 3.6, 1.7, 1.2, "ResNet-18\n(ImageNet, frozen)\n512 x 7 x 7", V); arr(5.5, 4.2, 5.8, 4.2)
box(5.8, 3.6, 2.0, 1.2, "Spatio-temporal graph\n28 x 49 nodes\nspatial 4-nbr + temporal +/-2\n2 residual GCN layers", V); arr(7.8, 4.2, 8.1, 4.2)
box(8.1, 3.6, 1.8, 1.2, "Factorised attention\nspatial a_s (7x7)\ntemporal a_t (28)", X); arr(9.9, 4.2, 10.2, 4.2)
box(10.2, 3.6, 1.3, 1.2, "Linear\nclassifier\np_video(PA4)", V)
box(8.1, 2.5, 1.8, 0.8, "Explanation:\n7x7 map + time curve", X); arr(9.0, 3.6, 9.0, 3.3)
ax.text(0.1, 2.1, "Wearable branch (physiology)", fontsize=10, weight="bold")
box(0.1, 0.4, 1.5, 1.2, "EDA / ECG\n512 Hz\n5.5 s window", W); arr(1.6, 1.0, 1.9, 1.0)
box(1.9, 0.4, 2.0, 1.2, "Hand-crafted features\nEDA 12, ECG 8\nper-participant z-score", W); arr(3.9, 1.0, 4.2, 1.0)
box(4.2, 0.4, 1.8, 1.2, "Logistic regression\n26 parameters\n0.36 ms / window", W)
arr(11.5, 4.2, 12.1, 3.0); arr(6.0, 1.0, 12.1, 2.4)
box(12.1, 2.1, 1.8, 1.2, "Late fusion\nmeta-classifier on\n(p_EDA, p_video)\n-> BL1 / PA4", F)
plt.savefig(os.path.join(a.out_dir, "fig_architecture.png"), dpi=220, bbox_inches="tight"); plt.close()
print("written:", sorted(os.listdir(a.out_dir)))
