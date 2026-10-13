"""
End-to-end latency and size of the contactless pipeline, per 5.5 s clip.

Times each stage separately on GPU and CPU (median of --reps runs after warm-up):
  detect   MTCNN on one full frame at 640 px (x n_frames for a clip)
  encode   backbone on n_frames aligned crops (resnet18 and mobilenet_v3_small)
  head     spatio-temporal graph head
Reports parameters and MB for each part. Head weights are random: latency does not
depend on trained values.

Run:
  python latency_v2.py --video_dir ../biovid/clips --crop_dir ../crops_v2 --out results_v2/latency.json
"""
import os, glob, json, time, argparse
import numpy as np, cv2, torch, torch.nn.functional as F, torchvision
from facenet_pytorch import MTCNN
from models_v2 import STGraph

p = argparse.ArgumentParser()
p.add_argument("--video_dir", required=True)
p.add_argument("--crop_dir", required=True)
p.add_argument("--n_frames", type=int, default=28)
p.add_argument("--reps", type=int, default=30)
p.add_argument("--cpu_threads", type=int, default=4)
p.add_argument("--out", default="latency.json")
a = p.parse_args()
torch.set_num_threads(a.cpu_threads)

f = sorted(glob.glob(os.path.join(a.crop_dir, "*.npz")))[0]
key = os.path.basename(f)[:-4]
vid = glob.glob(os.path.join(a.video_dir, "**", key + ".mp4"), recursive=True)[0]
cap = cv2.VideoCapture(vid); _, fr = cap.read(); cap.release()
frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
crops = np.load(f)["crops"][np.linspace(0, 27, a.n_frames).round().astype(int)]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def bench(fn, dev):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(a.reps):
        if dev == "cuda": torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        if dev == "cuda": torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def backbone(name):
    if name == "resnet18":
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        return torch.nn.Sequential(*list(m.children())[:-2]), 512
    m = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    return m.features, 576


npar = lambda m: sum(q.numel() for q in m.parameters())
res = {"n_frames": a.n_frames, "cpu_threads": a.cpu_threads,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "stages_ms": {}}
devs = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
for dev in devs:
    mt = MTCNN(keep_all=True, device=dev)
    sc = 640 / frame.shape[1]; small = cv2.resize(frame, None, fx=sc, fy=sc)
    det = bench(lambda: mt.detect(small, landmarks=True), dev)
    x = torch.from_numpy(crops).permute(0, 3, 1, 2).float() / 255
    x = ((F.interpolate(x, size=224, mode="bilinear", align_corners=False) - MEAN) / STD).to(dev)
    for name in ("resnet18", "mobilenet_v3_small"):
        bb, C = backbone(name); bb = bb.eval().to(dev)
        head = STGraph(C=C, T=a.n_frames, H=128, win=2).eval().to(dev)
        with torch.no_grad():
            fm = bb(x).flatten(2).transpose(1, 2)[None]
            enc = bench(lambda: bb(x), dev)
            hd = bench(lambda: head(fm), dev)
        r = {"detect_per_frame": det, "detect_clip": det * a.n_frames, "encode_clip": enc, "head_clip": hd,
             "total_clip": det * a.n_frames + enc + hd,
             "total_excluding_detection": enc + hd,
             "params_backbone": npar(bb), "params_head": npar(head),
             "size_MB_fp32": round((npar(bb) + npar(head)) * 4 / 1e6, 1)}
        res["stages_ms"][f"{dev}/{name}"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}
        print(dev, name, json.dumps(res["stages_ms"][f"{dev}/{name}"]), flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print("saved", a.out)
