# LightViT Project

This scaffold organizes LightViT compression experiments around the constraints below.

## Constraints

1. Preserve competitive top-1 accuracy after compression.
2. Reduce parameter count and checkpoint size through factorization.
3. Improve inference latency with lightweight architectural changes.
4. Support QAT-ready export paths for final deployment experiments.

## Layout

- `data/`: Dataset downloads and preprocessing artifacts.
- `checkpoints/`: Baseline, pure-SVD sweep, and QAT-stage checkpoints.
- `models/`: Baseline and compressed model definitions plus custom layers.
- `core/`: Calibration, factorization, and measurement utilities.
- `scripts/`: Ordered entry points for the training and evaluation pipeline.

## Target Metrics

- Baseline checkpoint: `checkpoints/baseline_fp32.pth`
- Pure-SVD sweep: `checkpoints/pure_svd/manifest.csv`
- Act-SVD sweep: `checkpoints/act_svd/manifest.csv`
- Hybrid sweep: `checkpoints/hybrid/manifest.csv`
- Hybrid HOOI sweep: `checkpoints/hybrid_hooi/manifest.csv`
- Hybrid Tensorly sweep: `checkpoints/hybrid_tensorly/manifest.csv`
- Final QAT checkpoint: `checkpoints/final_qat_int8.pth`
- Accuracy target: keep the compressed model within an acceptable regression budget.
- Compression target: lower parameter count and disk footprint versus the baseline.
- Speed target: measure end-to-end latency with repeatable warmup and timed runs.

## Suggested Workflow

1. Install dependencies from `requirements.txt`.
2. Run `scripts/01_train_baseline.py`.
3. Run `scripts/02_calibrate_svd.py` to create FP32 pure-SVD rank sweeps and `checkpoints/pure_svd/manifest.csv`.
4. Run `scripts/02a_calibrate_act_svd.py` to create FP32 Act-SVD rank sweeps and `checkpoints/act_svd/manifest.csv`.
5. Run `scripts/02c_calibrate_hybrid.py` to create FP32 hybrid Tucker rank sweeps and `checkpoints/hybrid/manifest.csv`.
6. Run `scripts/02d_calibrate_hybrid_hooi.py` to create FP32 hybrid Tucker-HOOI rank sweeps and `checkpoints/hybrid_hooi/manifest.csv`.
7. Run `scripts/02d_calibrate_hybrid_hooi.py --tucker-backend tensorly` to create Tensorly-optimized Tucker sweeps and `checkpoints/hybrid_tensorly/manifest.csv`.
8. Run `scripts/02b_recover_svd.py` on selected low-rank checkpoints to write recovered checkpoints and recovery metrics CSVs.
9. Run later QAT/INT8 scripts after wiring them to the recovered checkpoint.
