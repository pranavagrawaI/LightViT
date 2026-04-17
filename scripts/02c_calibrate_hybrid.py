"""Create a hybrid FP32 sweep: Act-SVD MLPs plus Tucker-factorized QKV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.calibration import CovarianceCalibrator
from core.metrics import count_parameters, measure_latency_ms, measure_model_size_mb
from models.baseline_vit import LightViTBaseline
from models.tucker_vit import HybridTuckerLightViT


DEFAULT_RANK_RATIOS = (0.125, 0.1875, 0.25, 0.375, 0.5)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "baseline_fp32.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "hybrid",
    )
    parser.add_argument(
        "--rank-ratios", type=float, nargs="+", default=DEFAULT_RANK_RATIOS
    )
    parser.add_argument("--head-rank", type=int, default=None)
    parser.add_argument("--calibration-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--eval-accuracy", action="store_true")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def ratio_slug(rank_ratio: float) -> str:
    return f"{rank_ratio:g}".replace(".", "p")


def checkpoint_size_mb(checkpoint_path: Path) -> float:
    return checkpoint_path.stat().st_size / (1024**2)


def load_baseline_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("state_dict", checkpoint)


def build_baseline_model(state_dict: dict[str, torch.Tensor]) -> LightViTBaseline:
    model = LightViTBaseline()
    model.load_state_dict(state_dict)
    return model


def build_hybrid_model(
    state_dict: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    rank_ratio: float,
    head_rank: int | None,
) -> HybridTuckerLightViT:
    model = HybridTuckerLightViT(rank_ratio=rank_ratio)
    model.load_state_dict(state_dict)
    model.apply_hybrid_tucker(covariances, rank_ratio, head_rank=head_rank)
    return model


def data_loader(
    train: bool, batch_size: int, num_workers: int
) -> torch.utils.data.DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    dataset = torchvision.datasets.CIFAR100(
        root=PROJECT_ROOT / "data",
        train=train,
        download=True,
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


@torch.inference_mode()
def collect_covariances(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    calibrator = CovarianceCalibrator()
    calibrator.register(model, module_types=(nn.Linear,))
    model.eval()
    progress = tqdm(loader, desc="Calibrating Hybrid MLPs", leave=False)
    for batch_idx, (images, _targets) in enumerate(progress, start=1):
        model(images.to(device, non_blocking=True))
        if batch_idx >= max_batches:
            break
    calibrator.remove()
    covariances = {
        name: covariance.detach().cpu()
        for name, covariance in calibrator.normalized_covariances().items()
    }
    return covariances, dict(calibrator.sample_counts)


@torch.inference_mode()
def measure_accuracy(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(inputs).argmax(dim=1)
        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()
        if max_batches is not None and batch_idx >= max_batches:
            break
    return 100.0 * correct / total


def collect_model_metrics(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader | None,
    skip_latency: bool,
    warmup_steps: int,
    timed_steps: int,
    max_eval_batches: int | None,
) -> dict[str, object]:
    params = count_parameters(model)
    model_size_mb = measure_model_size_mb(model)
    latency_ms: float | str = ""
    top1_accuracy: float | str = ""
    if not skip_latency or test_loader is not None:
        model = model.to(device)
    if not skip_latency:
        sample = torch.randn(1, 3, 32, 32, device=device)
        latency_ms = measure_latency_ms(
            model,
            sample,
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
        )
    if test_loader is not None:
        top1_accuracy = measure_accuracy(model, test_loader, device, max_eval_batches)
    model = model.cpu()
    return {
        "params": params,
        "model_size_mb": model_size_mb,
        "checkpoint_mb": checkpoint_size_mb(checkpoint_path),
        "latency_ms": latency_ms,
        "top1_accuracy": top1_accuracy,
    }


def format_metrics_row(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "params": metrics["params"],
        "model_size_mb": f"{metrics['model_size_mb']:.4f}",
        "checkpoint_mb": f"{metrics['checkpoint_mb']:.4f}",
        "latency_ms": f"{metrics['latency_ms']:.4f}"
        if metrics["latency_ms"] != ""
        else "",
        "top1_accuracy": f"{metrics['top1_accuracy']:.4f}"
        if metrics["top1_accuracy"] != ""
        else "",
    }


def save_manifest(manifest_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "stage",
        "family",
        "rank_ratio",
        "checkpoint",
        "source_checkpoint",
        "params",
        "model_size_mb",
        "checkpoint_mb",
        "latency_ms",
        "top1_accuracy",
        "calibration_batches",
        "calibration_keys",
        "head_rank",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if any(not 0.0 < ratio <= 1.0 for ratio in args.rank_ratios):
        raise ValueError("All rank ratios must be in the interval (0, 1].")
    if args.calibration_batches <= 0:
        raise ValueError("calibration-batches must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = load_baseline_state_dict(args.baseline_checkpoint)
    calibration_loader = data_loader(True, args.batch_size, args.num_workers)
    test_loader = (
        data_loader(False, args.batch_size, args.num_workers)
        if args.eval_accuracy
        else None
    )
    rows: list[dict[str, object]] = []

    print(f"[*] Baseline: {args.baseline_checkpoint}")
    print(f"[*] Output:   {args.output_dir}")
    print(f"[*] Device:   {device}")
    print(f"[*] Calibration batches: {args.calibration_batches}")

    calibration_model = build_baseline_model(state_dict).to(device)
    covariances, sample_counts = collect_covariances(
        calibration_model, calibration_loader, device, args.calibration_batches
    )
    print(f"[*] Captured {len(covariances)} covariance tensors")

    print("[*] Auditing baseline")
    baseline_model = build_baseline_model(state_dict)
    baseline_metrics = collect_model_metrics(
        baseline_model,
        args.baseline_checkpoint,
        device,
        test_loader,
        args.skip_latency,
        args.warmup_steps,
        args.timed_steps,
        args.max_eval_batches,
    )
    rows.append(
        {
            "stage": "baseline",
            "family": "baseline",
            "rank_ratio": 1.0,
            "checkpoint": str(args.baseline_checkpoint),
            "source_checkpoint": "",
            **format_metrics_row(baseline_metrics),
            "calibration_batches": args.calibration_batches,
            "calibration_keys": len(covariances),
            "head_rank": args.head_rank or "full",
        }
    )

    for rank_ratio in args.rank_ratios:
        print(f"[*] Building Hybrid rank_ratio={rank_ratio:g}")
        model = build_hybrid_model(state_dict, covariances, rank_ratio, args.head_rank)
        checkpoint_name = f"hybrid_rank_ratio_{ratio_slug(rank_ratio)}.pth"
        checkpoint_path = args.output_dir / checkpoint_name
        torch.save(
            {
                "state_dict": model.state_dict(),
                "compression": {
                    "family": "hybrid_tucker",
                    "rank_ratio": rank_ratio,
                    "source_checkpoint": str(args.baseline_checkpoint),
                    "calibration_batches": args.calibration_batches,
                    "calibration_keys": sorted(covariances.keys()),
                    "calibration_sample_counts": sample_counts,
                    "qkv_factorization": "tucker",
                    "mlp_factorization": "act_svd",
                    "out_proj_factorization": "pure_svd",
                    "head_rank": args.head_rank or "full",
                },
            },
            checkpoint_path,
        )
        metrics = collect_model_metrics(
            model,
            checkpoint_path,
            device,
            test_loader,
            args.skip_latency,
            args.warmup_steps,
            args.timed_steps,
            args.max_eval_batches,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint["metrics"] = metrics
        torch.save(checkpoint, checkpoint_path)
        rows.append(
            {
                "stage": "decomposed",
                "family": "hybrid_tucker",
                "rank_ratio": rank_ratio,
                "checkpoint": str(checkpoint_path),
                "source_checkpoint": str(args.baseline_checkpoint),
                **format_metrics_row(metrics),
                "calibration_batches": args.calibration_batches,
                "calibration_keys": len(covariances),
                "head_rank": args.head_rank or "full",
            }
        )

    manifest_path = args.output_dir / "manifest.csv"
    save_manifest(manifest_path, rows)
    print(f"[*] Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
