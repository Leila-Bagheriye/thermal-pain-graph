#!/bin/bash
# Gathers every new result into one folder: /scratch/big/home/leibag/pain_v2/results_v3
# Run on the server:  bash collect_v3.sh
P=/scratch/big/home/leibag/pain_v2
B=/scratch/big/home/leibag/biovid_bio
O=$P/results_v3
mkdir -p $O

# 1. new video runs (summary + per-participant lines)
for r in finetune_loso graph_mobile graph_f8 graph_f14; do
  cp $P/runs/$r/summary.json $O/${r}_summary.json 2>/dev/null || echo "missing summary: $r"
  cp $P/runs/$r/folds.jsonl  $O/${r}_folds.jsonl  2>/dev/null
  cp $P/runs/$r/config.json  $O/${r}_config.json  2>/dev/null
done

# 2. figures from explain_v2 (graph run) and the redrawn data figures
mkdir -p $O/figs_graph $O/figs_data
cp $P/runs/graph/figs/*      $O/figs_graph/ 2>/dev/null || echo "no graph figs"
cp $P/figs_data/*            $O/figs_data/  2>/dev/null || echo "no data figs"
[ -d $P/runs/finetune_loso/figs ] && mkdir -p $O/figs_finetune_loso && cp $P/runs/finetune_loso/figs/* $O/figs_finetune_loso/

# 3. latency
cp $P/results_v2/latency.json $O/ 2>/dev/null || echo "no latency.json"
cp $P/results_v2/bio_latency.json $O/ 2>/dev/null

# 4. physiology rerun
cp $B/bio_rerun.jsonl $O/ 2>/dev/null || echo "no bio_rerun.jsonl"
python - <<'PY' > $O/bio_rerun_summary.txt 2>&1
import json, collections, numpy as np
R = collections.defaultdict(list)
for l in open("/scratch/big/home/leibag/biovid_bio/bio_rerun.jsonl"):
    r = json.loads(l); R[r["tag"]].append(r)
for t, v in R.items():
    a = np.array([x["acc"] for x in v]); f = np.array([x["f1"] for x in v])
    print(f"{t}  folds={len(v)}  acc={a.mean():.4f} ± {a.std():.4f}  f1={f.mean():.4f}")
PY

# 5. software versions
python - <<'PY' > $O/versions.txt 2>&1
import sys, platform, torch, torchvision, numpy, scipy, sklearn, pandas, cv2, matplotlib
print("python", sys.version.split()[0]); print("platform", platform.platform())
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
for m in (torchvision, numpy, scipy, sklearn, pandas, cv2, matplotlib): print(m.__name__, m.__version__)
try:
    import importlib.metadata as md; print("facenet-pytorch", md.version("facenet-pytorch"))
except Exception as e: print("facenet-pytorch ?", e)
PY
lscpu | grep -E "Model name|^CPU\(s\)" >> $O/versions.txt

ls -R $O
