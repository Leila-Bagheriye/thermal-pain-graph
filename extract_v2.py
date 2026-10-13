"""
Stage 1 - face detection, alignment and cropping (BioVid Part A, BL1 vs PA4).

For every video it saves ONE file <out_dir>/<trial>.npz with
    crops : (T, S, S, 3) uint8  aligned face crops (RGB)
    ok    : (T,) bool           True where MTCNN found a face in that frame
    idx   : (T,) int            which video frames were sampled

Alignment: eyes levelled, crop centred between the eyes and the mouth,
side = 3.2 x eye-to-mouth distance (forehead to chin, mouth included).
Frames without a detection reuse the landmarks of the nearest detected frame.

Run:
  python extract_v2.py --video_dir /path/to/video --out_dir /scratch/big/home/leibag/crops_v2
Safe to stop and restart: finished trials are skipped.
"""
import os, glob, json, argparse
import numpy as np, cv2, torch
from facenet_pytorch import MTCNN

p = argparse.ArgumentParser()
p.add_argument("--video_dir", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--classes", default="BL1,PA4")
p.add_argument("--n_frames", type=int, default=28)
p.add_argument("--size", type=int, default=112)
p.add_argument("--det_width", type=int, default=640)
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=dev)
CLS = a.classes.split(",")


def align(rgb, lm, S):
    le, re, _, ml, mr = lm
    eye, mouth = (le + re) / 2, (ml + mr) / 2
    ang = np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))
    c = (eye + mouth) / 2
    side = 3.2 * np.linalg.norm(mouth - eye)
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, S / side)
    M[0, 2] += S / 2 - c[0]
    M[1, 2] += S / 2 - c[1]
    return cv2.warpAffine(rgb, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


vids = sorted(f for f in glob.glob(os.path.join(a.video_dir, "**", "*.mp4"), recursive=True)
              if any(f"-{c}-" in os.path.basename(f) for c in CLS))
print(f"videos found: {len(vids)}  device: {dev}", flush=True)

st = {"saved": 0, "skipped": 0, "unreadable": 0, "no_face": 0, "frames_missing": 0}
for i, f in enumerate(vids):
    key = os.path.basename(f).rsplit(".", 1)[0]
    out = os.path.join(a.out_dir, key + ".npz")
    if os.path.exists(out):
        st["skipped"] += 1
        continue
    cap, frames = cv2.VideoCapture(f), []
    while True:
        r, fr = cap.read()
        if not r:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        st["unreadable"] += 1
        print("unreadable:", f)
        continue

    idx = np.linspace(0, len(frames) - 1, a.n_frames).round().astype(int)
    rgbs, lms = [], [None] * a.n_frames
    for j, k in enumerate(idx):
        rgb = cv2.cvtColor(frames[k], cv2.COLOR_BGR2RGB)
        rgbs.append(rgb)
        sc = a.det_width / rgb.shape[1]
        boxes, _, pts = mtcnn.detect(cv2.resize(rgb, None, fx=sc, fy=sc), landmarks=True)
        if boxes is not None:
            b = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))  # largest face
            lms[j] = pts[b] / sc

    ok = np.array([l is not None for l in lms])
    if not ok.any():
        st["no_face"] += 1
        print("no face in any frame:", key)
        continue
    st["frames_missing"] += int((~ok).sum())
    good = np.where(ok)[0]
    crops = np.zeros((a.n_frames, a.size, a.size, 3), np.uint8)
    for j in range(a.n_frames):
        lm = lms[j] if ok[j] else lms[good[np.argmin(np.abs(good - j))]]
        crops[j] = align(rgbs[j], lm, a.size)
    np.savez_compressed(out, crops=crops, ok=ok, idx=idx)
    st["saved"] += 1
    if i % 50 == 0:
        print(f"{i}/{len(vids)} {st}", flush=True)

print("done:", st)
json.dump(st, open(os.path.join(a.out_dir, "_extract_stats.json"), "w"), indent=2)
