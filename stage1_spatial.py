"""
Stage 1 (spatial): align whole face, keep ResNet spatial feature map.
Output per trial: (T, 512, 7, 7) float16  -> ~200 KB, same as before.
Usage: python stage1_spatial.py 071309_w_21 --device cuda:0
"""
import argparse, glob, os
import cv2, numpy as np, torch
from facenet_pytorch import MTCNN
from torchvision.models import resnet18, ResNet18_Weights

STRIDE, SIZE, DETECT_W = 5, 224, 640


def build_encoder(device):
    net = resnet18(weights=ResNet18_Weights.DEFAULT)
    # keep everything up to, but NOT including, global average pooling
    trunk = torch.nn.Sequential(*list(net.children())[:-2])
    trunk = trunk.eval().to(device)
    for p in trunk.parameters():
        p.requires_grad = False
    return trunk


def align_face(rgb, kp, size=SIZE):
    """Level by eye centres, crop a face-proportional box."""
    le, re = np.array(kp[0]), np.array(kp[1])
    ang = np.degrees(np.arctan2(re[1]-le[1], re[0]-le[0]))
    ec = ((le[0]+re[0])/2.0, (le[1]+re[1])/2.0)
    M = cv2.getRotationMatrix2D(ec, ang, 1.0)
    rot = cv2.warpAffine(rgb, M, (rgb.shape[1], rgb.shape[0]))
    iod = float(np.linalg.norm(re-le)) or 1.0
    half = int(iod * 1.8)
    x0, y0 = int(max(0, ec[0]-half)), int(max(0, ec[1]-half*0.85))
    x1, y1 = int(min(rgb.shape[1], ec[0]+half)), int(min(rgb.shape[0], ec[1]+half))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(rot[y0:y1, x0:x1], (size, size))


def process(vp, det):
    cap = cv2.VideoCapture(vp)
    faces, i, miss, n = [], 0, 0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % STRIDE == 0:
            n += 1
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            H = int(rgb.shape[0] * DETECT_W / rgb.shape[1])
            boxes, _, lms = det.detect(cv2.resize(rgb, (DETECT_W, H)), landmarks=True)
            if boxes is not None:
                s = rgb.shape[1] / float(DETECT_W)
                kp = [(p[0]*s, p[1]*s) for p in lms[0]]
                faces.append(align_face(rgb, kp))
            else:
                miss += 1
        i += 1
    cap.release()
    return (np.stack(faces).astype(np.uint8) if faces else None), miss, n


@torch.no_grad()
def encode(faces, trunk, device):
    """(T,224,224,3) uint8 -> (T,512,7,7) float16"""
    T = faces.shape[0]
    x = torch.from_numpy(faces).float().div(255).sub(0.5).div(0.5)
    x = x.permute(0, 3, 1, 2).to(device)          # (T,3,224,224)
    f = trunk(x)                                   # (T,512,7,7)
    return f.half().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("--workdir", default=os.path.expanduser("~/biovid"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"subject {args.subject} | device {device}")
    trunk = build_encoder(device)
    det = MTCNN(keep_all=False, post_process=False, device=device, select_largest=True)

    src = f"{args.workdir}/clips/{args.subject}"
    out = f"{args.workdir}/feats_spatial"
    os.makedirs(out, exist_ok=True)

    vids = sorted(glob.glob(f"{src}/*.mp4"))
    print(f"clips found: {len(vids)}")
    for vp in vids:
        key = os.path.basename(vp)[:-4]
        fp = f"{out}/{key}_map.npy"
        if os.path.exists(fp):
            continue
        faces, miss, n = process(vp, det)
        if faces is None:
            print(f"  FAIL {key}")
            continue
        np.save(fp, encode(faces, trunk, device))
        print(f"  {key}: {faces.shape[0]} frames, missed {miss}/{n}")
    print("done")


if __name__ == "__main__":
    main()
