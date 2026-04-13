"""Recover a pure-SVD checkpoint with baseline-teacher distillation."""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.amp import GradScaler, autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.metrics import count_parameters, measure_latency_ms, measure_model_size_mb
from models.baseline_vit import LightViTBaseline
from models.compressed_vit import CompressedLightViT


CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "baseline_fp32.pth",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "checkpoints"
        / "pure_svd"
        / "pure_svd_rank_ratio_0p1875.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "pure_svd_recovered",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 2))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hard-target-weight", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--latency-warmup-steps", type=int, default=5)
    parser.add_argument("--latency-timed-steps", type=int, default=20)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("state_dict", checkpoint)


def checkpoint_compression(checkpoint_path: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("compression", {})


def checkpoint_size_mb(checkpoint_path: Path) -> float:
    if not checkpoint_path.exists():
        return 0.0
    return checkpoint_path.stat().st_size / (1024**2)


def rank_ratio_from_checkpoint(checkpoint_path: Path) -> float:
    compression = checkpoint_compression(checkpoint_path)
    if "rank_ratio" not in compression:
        raise ValueError(
            "Student checkpoint must include compression.rank_ratio metadata."
        )
    return float(compression["rank_ratio"])


def build_teacher(checkpoint_path: Path, device: torch.device) -> LightViTBaseline:
    teacher = LightViTBaseline().to(device)
    teacher.load_state_dict(checkpoint_state_dict(checkpoint_path))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def build_student(checkpoint_path: Path, device: torch.device) -> CompressedLightViT:
    rank_ratio = rank_ratio_from_checkpoint(checkpoint_path)
    student = CompressedLightViT(rank_ratio=rank_ratio)
    student.apply_pure_svd(rank_ratio)
    student.load_state_dict(checkpoint_state_dict(checkpoint_path))
    return student.to(device)


def data_loaders(
    batch_size: int, num_workers: int
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    trainset = torchvision.datasets.CIFAR100(
        root=PROJECT_ROOT / "data",
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = torchvision.datasets.CIFAR100(
        root=PROJECT_ROOT / "data",
        train=False,
        download=True,
        transform=transform_test,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, **loader_kwargs)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False, **loader_kwargs)
    return trainloader, testloader


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
        temperature**2
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    progress = tqdm(loader, desc="Validation", leave=False)
    for batch_idx, (images, targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            predictions = model(images).argmax(dim=1)
        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()
        progress.set_postfix(acc=f"{100.0 * correct / total:.2f}%")
        if max_batches is not None and batch_idx >= max_batches:
            break
    return 100.0 * correct / total


def save_recovered_checkpoint(
    model: CompressedLightViT,
    checkpoint_path: Path,
    source_checkpoint: Path,
    teacher_checkpoint: Path,
    family: str,
    best_acc: float,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    payload = {
        "state_dict": state_dict,
        "compression": {
            "family": family,
            "rank_ratio": model.rank_ratio,
            "source_checkpoint": str(source_checkpoint),
            "teacher_checkpoint": str(teacher_checkpoint),
            "recovered": True,
        },
        "metrics": {
            "best_top1_accuracy": best_acc,
            "epoch": epoch,
            "params": count_parameters(model),
            "model_size_mb": measure_model_size_mb(model),
            "checkpoint_mb": 0.0,
        },
        "recovery": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "eta_min": args.eta_min,
            "temperature": args.temperature,
            "hard_target_weight": args.hard_target_weight,
        },
    }
    torch.save(payload, checkpoint_path)
    payload["metrics"]["checkpoint_mb"] = checkpoint_size_mb(checkpoint_path)
    torch.save(payload, checkpoint_path)


def format_metric(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def write_metrics_csv(metrics_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "stage",
        "family",
        "rank_ratio",
        "epoch",
        "checkpoint",
        "source_checkpoint",
        "teacher_checkpoint",
        "params",
        "model_size_mb",
        "checkpoint_mb",
        "latency_ms",
        "top1_accuracy",
        "train_loss",
        "kd_loss",
        "ce_loss",
        "lr",
        "temperature",
        "hard_target_weight",
        "saved_checkpoint",
    ]
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_metric(row.get(key, "")) for key in fieldnames})


def model_eval_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    checkpoint_path: Path,
    max_batches: int | None,
    skip_latency: bool,
    latency_warmup_steps: int,
    latency_timed_steps: int,
) -> dict[str, object]:
    latency_ms: float | str = ""
    if not skip_latency:
        sample = torch.randn(1, 3, 32, 32, device=device)
        latency_ms = measure_latency_ms(
            model,
            sample,
            warmup_steps=latency_warmup_steps,
            timed_steps=latency_timed_steps,
        )
    return {
        "params": count_parameters(model),
        "model_size_mb": measure_model_size_mb(model),
        "checkpoint_mb": checkpoint_size_mb(checkpoint_path),
        "latency_ms": latency_ms,
        "top1_accuracy": evaluate(model, loader, device, max_batches),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.hard_target_weight <= 1.0:
        raise ValueError("hard-target-weight must be in [0, 1].")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = data_loaders(args.batch_size, args.num_workers)
    teacher = build_teacher(args.teacher_checkpoint, device)
    student = build_student(args.student_checkpoint, device)
    student_family = str(
        checkpoint_compression(args.student_checkpoint).get("family", "low_rank_svd")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.student_checkpoint.stem}_recovered.pth"
    metrics_path = args.output_dir / f"{args.student_checkpoint.stem}_recovery_metrics.csv"
    optimizer = optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )
    scaler = GradScaler(device=device.type, enabled=device.type == "cuda")
    ce_loss = nn.CrossEntropyLoss()
    rows: list[dict[str, object]] = []

    teacher_metrics = model_eval_metrics(
        teacher,
        testloader,
        device,
        args.teacher_checkpoint,
        args.max_val_batches,
        args.skip_latency,
        args.latency_warmup_steps,
        args.latency_timed_steps,
    )
    rows.append(
        {
            "stage": "teacher_baseline",
            "family": "baseline",
            "rank_ratio": 1.0,
            "epoch": 0,
            "checkpoint": str(args.teacher_checkpoint),
            "source_checkpoint": "",
            "teacher_checkpoint": "",
            "params": teacher_metrics["params"],
            "model_size_mb": teacher_metrics["model_size_mb"],
            "checkpoint_mb": teacher_metrics["checkpoint_mb"],
            "latency_ms": teacher_metrics["latency_ms"],
            "top1_accuracy": teacher_metrics["top1_accuracy"],
            "train_loss": "",
            "kd_loss": "",
            "ce_loss": "",
            "lr": "",
            "temperature": "",
            "hard_target_weight": "",
            "saved_checkpoint": False,
        }
    )

    student_metrics = model_eval_metrics(
        student,
        testloader,
        device,
        args.student_checkpoint,
        args.max_val_batches,
        args.skip_latency,
        args.latency_warmup_steps,
        args.latency_timed_steps,
    )
    rows.append(
        {
            "stage": "pre_recovery",
            "family": student_family,
            "rank_ratio": student.rank_ratio,
            "epoch": 0,
            "checkpoint": str(args.student_checkpoint),
            "source_checkpoint": str(args.student_checkpoint),
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "params": student_metrics["params"],
            "model_size_mb": student_metrics["model_size_mb"],
            "checkpoint_mb": student_metrics["checkpoint_mb"],
            "latency_ms": student_metrics["latency_ms"],
            "top1_accuracy": student_metrics["top1_accuracy"],
            "train_loss": "",
            "kd_loss": "",
            "ce_loss": "",
            "lr": "",
            "temperature": args.temperature,
            "hard_target_weight": args.hard_target_weight,
            "saved_checkpoint": False,
        }
    )
    write_metrics_csv(metrics_path, rows)

    best_acc = -1.0
    print(f"[*] Student checkpoint: {args.student_checkpoint}")
    print(f"[*] Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"[*] Output checkpoint:  {output_path}")
    print(f"[*] Metrics CSV:         {metrics_path}")
    print(f"[*] Teacher accuracy:    {teacher_metrics['top1_accuracy']:.2f}%")
    print(f"[*] Starting student accuracy: {student_metrics['top1_accuracy']:.2f}%")

    for epoch in range(args.epochs):
        student.train()
        epoch_lr = optimizer.param_groups[0]["lr"]
        running_total = 0.0
        running_kd = 0.0
        running_ce = 0.0
        progress = tqdm(
            trainloader, desc=f"Epoch {epoch + 1}/{args.epochs} [recover]", leave=False
        )

        for step, (images, targets) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad(), autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                teacher_logits = teacher(images)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                student_logits = student(images)
                kd = distillation_loss(
                    student_logits, teacher_logits, args.temperature
                )
                ce = ce_loss(student_logits, targets)
                loss = (
                    1.0 - args.hard_target_weight
                ) * kd + args.hard_target_weight * ce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running_total += loss.item()
            running_kd += kd.item()
            running_ce += ce.item()
            progress.set_postfix(
                loss=f"{running_total / step:.4f}",
                kd=f"{running_kd / step:.4f}",
                ce=f"{running_ce / step:.4f}",
            )
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break

        scheduler.step()
        epoch_metrics = model_eval_metrics(
            student,
            testloader,
            device,
            output_path,
            args.max_val_batches,
            args.skip_latency,
            args.latency_warmup_steps,
            args.latency_timed_steps,
        )
        acc = float(epoch_metrics["top1_accuracy"])
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"Loss: {running_total / step:.4f} | "
            f"KD: {running_kd / step:.4f} | "
            f"CE: {running_ce / step:.4f} | "
            f"Val Acc: {acc:.2f}%"
        )

        saved_checkpoint = False
        if acc > best_acc:
            best_acc = acc
            save_recovered_checkpoint(
                student,
                output_path,
                args.student_checkpoint,
                args.teacher_checkpoint,
                student_family,
                best_acc,
                epoch + 1,
                args,
            )
            epoch_metrics["checkpoint_mb"] = checkpoint_size_mb(output_path)
            saved_checkpoint = True
            print(f"[*] Saved recovered checkpoint ({best_acc:.2f}%)")

        rows.append(
            {
                "stage": "recovery_epoch",
                "family": student_family,
                "rank_ratio": student.rank_ratio,
                "epoch": epoch + 1,
                "checkpoint": str(output_path) if saved_checkpoint else "",
                "source_checkpoint": str(args.student_checkpoint),
                "teacher_checkpoint": str(args.teacher_checkpoint),
                "params": epoch_metrics["params"],
                "model_size_mb": epoch_metrics["model_size_mb"],
                "checkpoint_mb": epoch_metrics["checkpoint_mb"]
                if saved_checkpoint
                else "",
                "latency_ms": epoch_metrics["latency_ms"],
                "top1_accuracy": epoch_metrics["top1_accuracy"],
                "train_loss": running_total / step,
                "kd_loss": running_kd / step,
                "ce_loss": running_ce / step,
                "lr": epoch_lr,
                "temperature": args.temperature,
                "hard_target_weight": args.hard_target_weight,
                "saved_checkpoint": saved_checkpoint,
            }
        )
        write_metrics_csv(metrics_path, rows)

    print(f"[*] Recovery complete. Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
