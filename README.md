# LightViT Project

This scaffold organizes a four-phase LightViT compression pipeline around the constraints below.

## Constraints

1. Preserve competitive top-1 accuracy after compression.
2. Reduce parameter count and checkpoint size through factorization.
3. Improve inference latency with lightweight architectural changes.
4. Support QAT-ready export paths for final deployment experiments.

## Layout

- `data/`: Dataset downloads and preprocessing artifacts.
- `checkpoints/`: Baseline, factorized, and QAT-stage checkpoints.
- `models/`: Baseline and compressed model definitions plus custom layers.
- `core/`: Calibration, factorization, and measurement utilities.
- `scripts/`: Ordered entry points for the training and evaluation pipeline.

## Target Metrics

- Baseline checkpoint: `checkpoints/baseline_fp32.pth`
- Factorized checkpoint: `checkpoints/factorized_fp32.pth`
- Final QAT checkpoint: `checkpoints/final_qat_int8.pth`
- Accuracy target: keep the compressed model within an acceptable regression budget.
- Compression target: lower parameter count and disk footprint versus the baseline.
- Speed target: measure end-to-end latency with repeatable warmup and timed runs.

## Suggested Workflow

1. Install dependencies from `requirements.txt`.
2. Run `scripts/01_train_baseline.py`.
3. Run `scripts/02_calibrate_svd.py`.
4. Run `scripts/03_train_qat.py`.
5. Run `scripts/04_evaluate.py`.