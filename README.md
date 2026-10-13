# Thermal Pain Graph

Code for *Explainable spatio-temporal graph learning for subject-independent pain detection* (CVC 2027).
Binary detection of acute experimental heat pain (BL1 vs PA4) on the BioVid Heat Pain Database, Part A.
The data are not included; request access from the BioVid authors.

## Pipeline

| Script | Purpose |
|---|---|
| `extract_v2.py` | MTCNN face detection and alignment, 28 frames per trial, 112 x 112 crops |
| `models_v2.py` | Spatio-temporal graph with factorised spatial and temporal attention |
| `stgraph_v2.py` | Frozen-encoder models under leave-one-subject-out: `graph`, `nograph`, `linear`; `--shuffle_labels 1` leakage control; faithfulness test |
| `finetune_v2.py` | Encoder fine-tuning; `--loso 1` for leave-one-subject-out, otherwise 5-fold subject-grouped CV |
| `train_bio_loso.py` | Skin-conductance / ECG / EMG features and classifiers under leave-one-subject-out |
| `fusion_v2.py` | Late fusion of wearable and video probabilities |
| `explain_v2.py` | Spatial, temporal, example and faithfulness figures |
| `fig_data_v2.py` | Data and architecture figures |
| `latency_v2.py` | Per-stage latency and model size on GPU and CPU |

## Reproduce

```bash
pip install -r requirements.txt
python extract_v2.py --video_dir <BioVid>/video --out_dir crops_v2
python stgraph_v2.py --crop_dir crops_v2 --model graph   --out_dir runs/graph
python stgraph_v2.py --crop_dir crops_v2 --model nograph --out_dir runs/nograph
python stgraph_v2.py --crop_dir crops_v2 --model linear  --out_dir runs/linear
python stgraph_v2.py --crop_dir crops_v2 --model graph   --out_dir runs/shuffled --shuffle_labels 1
python finetune_v2.py --crop_dir crops_v2 --out_dir runs/finetune_loso --loso 1
python train_bio_loso.py --root <BioVid>/biosignals_filtered --mods gsr --model lr
python fusion_v2.py --bio_feats feats_BL1_PA4.csv --video_run runs/graph --out results/fusion_graph.json
python explain_v2.py --run_dir runs/graph --crop_dir crops_v2
python latency_v2.py --video_dir <BioVid>/video --crop_dir crops_v2 --out results/latency.json
```

Every run writes `config.json`, one line per held-out participant in `folds.jsonl`, and `summary.json`. Runs restart where they stopped.

## Results

`results/` holds the summaries and per-participant results behind every table in the paper.

## License

See `LICENSE.txt`.
