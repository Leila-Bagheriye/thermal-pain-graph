# Thermal Pain Graph — spatio-temporal graph learning for facial pain assessment

Reference implementation for:

> **Explainable Spatio-Temporal Graph Learning for Facial Pain Assessment**
> Leila Bagheriye, Johan Kwisthout — Donders Institute, Radboud University

The model treats a trial as a sequence of whole-face frames and builds a
spatio-temporal graph whose nodes are spatial cells of the face at each time
step. An attention head over those nodes produces a 7 x 7 importance map inside
the forward pass, so the explanation is an architectural output rather than a
post-hoc approximation.

## Pipeline

Two stages, run in order.

**Stage 1 — feature extraction** (`stage1_spatial.py`)

Decodes each trial video, samples every fifth frame (28 frames per trial), runs
MTCNN at 640 px to detect the face and five landmarks, aligns by rotating the
inter-ocular axis horizontal, crops a square region of side 1.8 x inter-ocular
distance centred on the eye midpoint, resizes to 224 x 224, and pushes each
frame through a frozen ImageNet ResNet-18 trunk truncated before global average
pooling. Output per trial is a `(T, 512, 7, 7)` float16 array saved as
`<participant>-<CLASS>-<trial>_map.npy` (~200 KB per trial).

**Stage 2 — graph model and LOSO evaluation** (`stage2_spatial.py`)

Loads the cached feature maps, builds a graph over 28 x 49 = 1,372 nodes, applies
two row-normalised graph-convolutional layers with a learned attention head, and
evaluates under leave-one-subject-out cross-validation.

## Data

This work uses the **BioVid Heat Pain Database, Part A** (Walter et al., 2013).
The database is available from its authors under a data-use agreement and is
**not redistributed here** — neither the raw videos nor the derived feature
caches. To reproduce the results, obtain the database yourself and arrange the
clips as `clips/<participant>/<participant>-<CLASS>-<trial>.mp4`, with `CLASS` in
`BL1, PA1, PA2, PA3, PA4`.

Note that BioVid is de-identified facial video; please respect the database's
terms of use and do not redistribute frames or derived embeddings.

## Requirements

```
pip install -r requirements.txt
```

Stage 1 needs a CUDA-capable GPU for reasonable throughput (`facenet-pytorch`
for MTCNN, `torchvision` for ResNet-18). Stage 2 runs on CPU but is far faster on
GPU.

## Running

Extract features, one subject at a time:

```bash
python stage1_spatial.py 071309_w_21 --device cuda:0
```

Evaluate the binary task (BL1 vs PA4) under LOSO:

```bash
python stage2_spatial.py --binary --epochs 40 --frames 28 --lr 5e-4 --device cuda:0
```

Evaluate all five classes instead, omitting `--binary`:

```bash
python stage2_spatial.py --epochs 40 --frames 28 --lr 5e-4 --device cuda:0
```

Useful flags: `--cache` (feature directory), `--out` (results directory),
`--max_folds N` (stop after N held-out subjects, for smoke tests),
`--batch`, `--epochs`.

## Outputs

Stage 2 writes into `--out` (default `~/biovid/st_results`):

| File | Contents |
| --- | --- |
| `fold_acc` rows in `folds.csv` | one line per completed fold: `<subject>,<accuracy>` |
| `st_spatial.json` | full result record: fold accuracies, mean, std, per-class P/R/F1, macro F1, 7 x 7 saliency map, confusion matrix |
| `saliency_map.npy` | the 7 x 7 mean attention map as an array |

The per-fold CSV is appended as the run proceeds, so it survives a crash. Clear
it before a fresh run if you want fold counts to refer to a single run.

## Reproducibility

Reported result: **0.5635 mean accuracy across 86 LOSO folds** (population std
0.1161), binary BL1 vs PA4, against 0.50 chance. Per-class: BL1 P/R/F1 =
0.565 / 0.553 / 0.559; PA4 = 0.562 / 0.574 / 0.568. Pooled confusion matrix:
951 true negatives, 768 false positives, 733 false negatives, 987 true
positives over 3,439 test trials.

The published run did **not** set a random seed. Two runs of the same command
will not reproduce the same fold array, because the per-fold model initialisation
and per-epoch shuffling both draw from global random state. If you need a
bit-identical rerun, set `torch.manual_seed(...)` and `np.random.seed(...)` near
the top of `main()` before launching.

`results/folds.csv` and `results/st_spatial.json` in this repository are the
artefacts of the published run.

## Citation

```bibtex
@article{bagheriye2027thermalpain,
  title   = {Explainable Spatio-Temporal Graph Learning for Facial Pain Assessment},
  author  = {Bagheriye, Leila and Kwisthout, Johan},
  year    = {2027},
  note    = {Donders Institute, Radboud University}
}
```

## License

Code released under the MIT License (see `LICENSE`). The BioVid database is
governed by its own terms and is not covered by this license.
