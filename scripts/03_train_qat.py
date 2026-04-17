"""Train selected recovered checkpoints with fixed-budget fake-quant QAT."""

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
from torch.ao.quantization import FakeQuantize
from torch.ao.quantization.fake_quantize import disable_observer
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.metrics import count_parameters, measure_latency_ms, measure_model_size_mb
from models.baseline_vit import LightViTBaseline
from models.compressed_vit import CompressedLightViT
from models.tucker_vit import HybridTuckerLightViT, TuckerSelfAttention


CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
DEFAULT_CHECKPOINTS = (
    PROJECT_ROOT
    / "checkpoints"
    / "pure_svd_recovered"
    / "pure_svd_rank_ratio_0p5_recovered.pth",
    PROJECT_ROOT
    / "checkpoints"
    / "hybrid_recovered"
    / "hybrid_rank_ratio_0p5_recovered.pth",
    PROJECT_ROOT
    / "checkpoints"
    / "hybrid_tensorly_recovered"
    / "hybrid_tensorly_rank_ratio_0p5_recovered.pth",
    PROJECT_ROOT
    / "checkpoints"
    / "pure_svd_recovered"
    / "pure_svd_rank_ratio_0p25_recovered.pth",
)
QAT_METRIC_FIELDNAMES = [
    "stage",
    "family",
    "rank_ratio",
    "epoch",
    "checkpoint",
    "source_checkpoint",
    "teacher_checkpoint",
    "params",
    "model_size_mb",
    "estimated_int8_model_size_mb",
    "checkpoint_mb",
    "latency_ms",
    "top1_accuracy",
    "source_pre_recovery_top1_accuracy",
    "train_loss",
    "kd_loss",
    "ce_loss",
    "lr",
    "temperature",
    "hard_target_weight",
    "qat_mode",
    "observer_enabled",
    "saved_checkpoint",
]


def activation_fake_quant() -> FakeQuantize:
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )


def weight_fake_quant(ch_axis: int = 0) -> FakeQuantize:
    return FakeQuantize(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=ch_axis,
    )


def tensor_weight_fake_quant() -> FakeQuantize:
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
    )


class QATLinear(nn.Module):
    def __init__(self, linear: nn.Linear, quantize_activations: bool) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.linear = linear
        self.quantize_activations = quantize_activations
        self.input_fake_quant = activation_fake_quant() if quantize_activations else nn.Identity()
        self.weight_fake_quant = weight_fake_quant(ch_axis=0)
        self.output_fake_quant = activation_fake_quant() if quantize_activations else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.input_fake_quant(inputs)
        weight = self.weight_fake_quant(self.linear.weight)
        outputs = F.linear(inputs, weight, self.linear.bias)
        return self.output_fake_quant(outputs)


class QATConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, quantize_activations: bool) -> None:
        super().__init__()
        self.conv = conv
        self.quantize_activations = quantize_activations
        self.input_fake_quant = activation_fake_quant() if quantize_activations else nn.Identity()
        self.weight_fake_quant = weight_fake_quant(ch_axis=0)
        self.output_fake_quant = activation_fake_quant() if quantize_activations else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.input_fake_quant(inputs)
        weight = self.weight_fake_quant(self.conv.weight)
        outputs = F.conv2d(
            inputs,
            weight,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )
        return self.output_fake_quant(outputs)


class QATTuckerSelfAttention(nn.Module):
    def __init__(
        self, attention: TuckerSelfAttention, quantize_activations: bool
    ) -> None:
        super().__init__()
        self.attention = attention
        self.quantize_activations = quantize_activations
        self.input_fake_quant = activation_fake_quant() if quantize_activations else nn.Identity()
        self.core_fake_quant = tensor_weight_fake_quant()
        self.factor_heads_fake_quant = tensor_weight_fake_quant()
        self.factor_out_fake_quant = tensor_weight_fake_quant()
        self.factor_in_fake_quant = tensor_weight_fake_quant()
        self.output_fake_quant = activation_fake_quant() if quantize_activations else nn.Identity()

    @property
    def embed_dim(self) -> int:
        return self.attention.embed_dim

    @property
    def num_heads(self) -> int:
        return self.attention.num_heads

    @property
    def batch_first(self) -> bool:
        return self.attention.batch_first

    @property
    def _qkv_same_embed_dim(self) -> bool:
        return True

    @property
    def in_proj_bias(self) -> None:
        return None

    @property
    def in_proj_weight(self) -> None:
        return None

    def _project_qkv(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "bli,ir,qhdr,Hh,Dd->blqHD",
            inputs,
            self.factor_in_fake_quant(self.attention.factor_in),
            self.core_fake_quant(self.attention.core),
            self.factor_heads_fake_quant(self.attention.factor_heads),
            self.factor_out_fake_quant(self.attention.factor_out),
        ) + self.attention.qkv_bias.view(
            1, 1, 3, self.attention.num_heads, self.attention.head_dim
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.attention.batch_first:
            query, key, value = (
                tensor.transpose(0, 1) for tensor in (query, key, value)
            )
        if query is not key or query is not value:
            raise ValueError("QATTuckerSelfAttention only supports self-attention.")

        query = self.input_fake_quant(query)
        qkv = self._project_qkv(query)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        mask = self.attention._additive_mask(
            attn_mask, key_padding_mask, query, key.shape[1]
        )

        if need_weights:
            outputs, weights = self.attention._attention_with_weights(
                q, k, v, mask, is_causal, average_attn_weights
            )
        else:
            outputs = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attention.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            weights = None

        outputs = outputs.transpose(1, 2).contiguous().view(
            query.shape[0], query.shape[1], self.attention.embed_dim
        )
        outputs = self.attention.out_proj(outputs)
        outputs = self.output_fake_quant(outputs)
        if not self.attention.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=list(DEFAULT_CHECKPOINTS),
        help="Recovered checkpoints to train with the same QAT budget.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "baseline_fp32.pth",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "qat"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 2))
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hard-target-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--qat-mode",
        choices=("weight-only", "weight-activation"),
        default="weight-only",
        help="weight-only is much faster; weight-activation also fake-quantizes activations.",
    )
    parser.add_argument("--disable-observer-after", type=int, default=1)
    parser.add_argument("--latency-warmup-steps", type=int, default=5)
    parser.add_argument("--latency-timed-steps", type=int, default=20)
    parser.add_argument("--skip-latency", action="store_true", default=True)
    parser.add_argument("--measure-latency", dest="skip_latency", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=20)
    parser.add_argument("--final-full-eval", action="store_true")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_payload(checkpoint_path: Path) -> dict[str, object]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint if isinstance(checkpoint, dict) else {"state_dict": checkpoint}


def checkpoint_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = checkpoint_payload(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no state_dict: {checkpoint_path}")
    return state_dict


def checkpoint_compression(checkpoint_path: Path) -> dict[str, object]:
    compression = checkpoint_payload(checkpoint_path).get("compression", {})
    return compression if isinstance(compression, dict) else {}


def reported_family(checkpoint_path: Path) -> str:
    if checkpoint_path.parent.name.startswith("hybrid_tensorly"):
        return "hybrid_tensorly"
    return str(checkpoint_compression(checkpoint_path).get("family", ""))


def checkpoint_metrics(checkpoint_path: Path) -> dict[str, object]:
    metrics = checkpoint_payload(checkpoint_path).get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def pre_recovery_accuracy(checkpoint_path: Path) -> float | str:
    metrics_path = checkpoint_path.with_name(
        checkpoint_path.stem.removesuffix("_recovered") + "_recovery_metrics.csv"
    )
    if not metrics_path.exists():
        return ""
    with metrics_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("stage") == "pre_recovery":
                value = row.get("top1_accuracy", "")
                return float(value) if value else ""
    return ""


def checkpoint_size_mb(checkpoint_path: Path) -> float:
    if not checkpoint_path.exists():
        return 0.0
    return checkpoint_path.stat().st_size / (1024**2)


def build_teacher(checkpoint_path: Path, device: torch.device) -> LightViTBaseline:
    teacher = LightViTBaseline().to(device)
    teacher.load_state_dict(checkpoint_state_dict(checkpoint_path))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def build_student(checkpoint_path: Path, device: torch.device) -> nn.Module:
    compression = checkpoint_compression(checkpoint_path)
    family = str(compression.get("family", "pure_svd"))
    if reported_family(checkpoint_path) == "hybrid_tensorly":
        family = "hybrid_tucker"
    rank_ratio = float(compression.get("rank_ratio", 1.0))
    if family == "hybrid_tucker":
        student = HybridTuckerLightViT(rank_ratio=rank_ratio)
        student.apply_hybrid_tucker(rank_ratio=rank_ratio)
    elif family in {"pure_svd", "act_svd", "low_rank_svd"}:
        student = CompressedLightViT(rank_ratio=rank_ratio)
        student.apply_pure_svd(rank_ratio)
    else:
        raise ValueError(f"Unsupported QAT student family: {family}")
    student.load_state_dict(checkpoint_state_dict(checkpoint_path))
    return student.to(device)


def replace_qat_modules(module: nn.Module, quantize_activations: bool) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QATLinear(child, quantize_activations))
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QATConv2d(child, quantize_activations))
            continue
        replace_qat_modules(child, quantize_activations)
        if isinstance(child, TuckerSelfAttention):
            setattr(
                module,
                name,
                QATTuckerSelfAttention(child, quantize_activations),
            )


def apply_qat(model: nn.Module, qat_mode: str = "weight-only") -> nn.Module:
    replace_qat_modules(model, quantize_activations=qat_mode == "weight-activation")
    return model


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
    max_batches: int | None,
    use_amp: bool,
) -> float:
    model.eval()
    correct = 0
    total = 0
    progress = tqdm(loader, desc="Validation", leave=False)
    for batch_idx, (images, targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            predictions = model(images).argmax(dim=1)
        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()
        progress.set_postfix(acc=f"{100.0 * correct / total:.2f}%")
        if max_batches is not None and batch_idx >= max_batches:
            break
    return 100.0 * correct / total


def model_eval_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    checkpoint_path: Path,
    max_batches: int | None,
    skip_latency: bool,
    latency_warmup_steps: int,
    latency_timed_steps: int,
    use_amp: bool,
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
        "top1_accuracy": evaluate(model, loader, device, max_batches, use_amp),
    }


def qat_module_count(model: nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if isinstance(module, (QATLinear, QATConv2d, QATTuckerSelfAttention))
    )


def quantizable_parameter_bytes(model: nn.Module) -> int:
    parameter_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, QATLinear):
            parameter_ids.add(id(module.linear.weight))
        elif isinstance(module, QATConv2d):
            parameter_ids.add(id(module.conv.weight))
        elif isinstance(module, QATTuckerSelfAttention):
            parameter_ids.update(
                {
                    id(module.attention.core),
                    id(module.attention.factor_heads),
                    id(module.attention.factor_out),
                    id(module.attention.factor_in),
                }
            )
    total = 0
    for parameter in model.parameters():
        total += parameter.numel() if id(parameter) in parameter_ids else parameter.numel() * parameter.element_size()
    return total


def estimated_int8_model_size_mb(model: nn.Module) -> float:
    param_size = quantizable_parameter_bytes(model)
    buffer_size = 0
    for name, buffer in model.named_buffers():
        if "fake_quant" in name or "activation_post_process" in name:
            continue
        buffer_size += buffer.numel() * buffer.element_size()
    return (param_size + buffer_size) / (1024**2)


def save_qat_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    source_checkpoint: Path,
    best_acc: float,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "compression": checkpoint_compression(source_checkpoint),
        "source_metrics": checkpoint_metrics(source_checkpoint),
        "qat": {
            "method": "fake_quant_qat",
            "qat_mode": args.qat_mode,
            "source_checkpoint": str(source_checkpoint),
            "source_family": reported_family(source_checkpoint),
            "epochs": args.epochs,
            "best_epoch": epoch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "eta_min": args.eta_min,
            "temperature": args.temperature,
            "hard_target_weight": args.hard_target_weight,
            "disable_observer_after": args.disable_observer_after,
            "qat_modules": qat_module_count(model),
        },
        "metrics": {
            "best_top1_accuracy": best_acc,
            "epoch": epoch,
            "params": count_parameters(model),
            "model_size_mb": measure_model_size_mb(model),
            "estimated_int8_model_size_mb": estimated_int8_model_size_mb(model),
            "checkpoint_mb": 0.0,
        },
    }
    torch.save(payload, checkpoint_path)
    payload["metrics"]["checkpoint_mb"] = checkpoint_size_mb(checkpoint_path)
    torch.save(payload, checkpoint_path)


def format_metric(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_metric(row.get(key, "")) for key in fieldnames})


def train_one_checkpoint(
    checkpoint_path: Path,
    teacher: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    source_metrics = checkpoint_metrics(checkpoint_path)
    compression = checkpoint_compression(checkpoint_path)
    family = reported_family(checkpoint_path)
    student = apply_qat(
        build_student(checkpoint_path, torch.device("cpu")), args.qat_mode
    ).to(device)
    output_path = args.output_dir / f"{checkpoint_path.stem}_qat_best.pth"
    last_path = args.output_dir / f"{checkpoint_path.stem}_qat_last.pth"
    metrics_path = args.output_dir / f"{checkpoint_path.stem}_qat_metrics.csv"
    use_amp = args.amp and device.type == "cuda"

    optimizer = optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )
    scaler = GradScaler(device=device.type, enabled=use_amp)
    ce_loss = nn.CrossEntropyLoss()
    rows: list[dict[str, object]] = []
    rank_ratio = compression.get("rank_ratio", "")
    source_pre_recovery_acc = pre_recovery_accuracy(checkpoint_path)

    pre_metrics = model_eval_metrics(
        student,
        testloader,
        device,
        checkpoint_path,
        args.max_val_batches,
        args.skip_latency,
        args.latency_warmup_steps,
        args.latency_timed_steps,
        use_amp,
    )
    rows.append(
        {
            "stage": "pre_qat",
            "family": family,
            "rank_ratio": rank_ratio,
            "epoch": 0,
            "checkpoint": str(checkpoint_path),
            "source_checkpoint": str(checkpoint_path),
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "params": pre_metrics["params"],
            "model_size_mb": pre_metrics["model_size_mb"],
            "estimated_int8_model_size_mb": estimated_int8_model_size_mb(student),
            "checkpoint_mb": pre_metrics["checkpoint_mb"],
            "latency_ms": pre_metrics["latency_ms"],
            "top1_accuracy": pre_metrics["top1_accuracy"],
            "source_pre_recovery_top1_accuracy": source_pre_recovery_acc,
            "train_loss": "",
            "kd_loss": "",
            "ce_loss": "",
            "lr": "",
            "temperature": args.temperature,
            "hard_target_weight": args.hard_target_weight,
            "qat_mode": args.qat_mode,
            "observer_enabled": True,
            "saved_checkpoint": False,
        }
    )
    write_csv(metrics_path, rows, QAT_METRIC_FIELDNAMES)

    best_acc = -1.0
    final_acc = float(pre_metrics["top1_accuracy"])
    print(f"[*] QAT source: {checkpoint_path}")
    print(f"[*] QAT output: {output_path}")
    print(f"[*] Starting accuracy: {final_acc:.2f}%")

    for epoch in range(args.epochs):
        if epoch >= args.disable_observer_after:
            student.apply(disable_observer)

        student.train()
        epoch_lr = optimizer.param_groups[0]["lr"]
        running_total = 0.0
        running_kd = 0.0
        running_ce = 0.0
        progress = tqdm(
            trainloader,
            desc=f"{checkpoint_path.stem} epoch {epoch + 1}/{args.epochs} [qat]",
            leave=False,
        )

        for step, (images, targets) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
                teacher_logits = teacher(images)

            with autocast(device_type=device.type, enabled=use_amp):
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
            use_amp,
        )
        final_acc = float(epoch_metrics["top1_accuracy"])
        saved_checkpoint = False
        if final_acc > best_acc:
            best_acc = final_acc
            save_qat_checkpoint(student, output_path, checkpoint_path, best_acc, epoch + 1, args)
            saved_checkpoint = True

        rows.append(
            {
                "stage": "qat_epoch",
                "family": family,
                "rank_ratio": rank_ratio,
                "epoch": epoch + 1,
                "checkpoint": str(output_path) if saved_checkpoint else "",
                "source_checkpoint": str(checkpoint_path),
                "teacher_checkpoint": str(args.teacher_checkpoint),
                "params": epoch_metrics["params"],
                "model_size_mb": epoch_metrics["model_size_mb"],
                "estimated_int8_model_size_mb": estimated_int8_model_size_mb(student),
                "checkpoint_mb": checkpoint_size_mb(output_path)
                if saved_checkpoint
                else "",
                "latency_ms": epoch_metrics["latency_ms"],
                "top1_accuracy": epoch_metrics["top1_accuracy"],
                "source_pre_recovery_top1_accuracy": source_pre_recovery_acc,
                "train_loss": running_total / step,
                "kd_loss": running_kd / step,
                "ce_loss": running_ce / step,
                "lr": epoch_lr,
                "temperature": args.temperature,
                "hard_target_weight": args.hard_target_weight,
                "qat_mode": args.qat_mode,
                "observer_enabled": epoch < args.disable_observer_after,
                "saved_checkpoint": saved_checkpoint,
            }
        )
        write_csv(metrics_path, rows, QAT_METRIC_FIELDNAMES)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"Loss: {running_total / step:.4f} | "
            f"Val Acc: {final_acc:.2f}% | "
            f"Best: {best_acc:.2f}%"
        )

    save_qat_checkpoint(student, last_path, checkpoint_path, final_acc, args.epochs, args)
    full_best_acc: float | str = ""
    full_final_acc: float | str = ""
    if args.final_full_eval:
        best_model = apply_qat(
            build_student(checkpoint_path, torch.device("cpu")), args.qat_mode
        ).to(device)
        best_payload = torch.load(output_path, map_location=device)
        best_model.load_state_dict(best_payload["state_dict"])
        full_best_acc = evaluate(best_model, testloader, device, None, use_amp)
        full_final_acc = evaluate(student, testloader, device, None, use_amp)

    return {
        "source_checkpoint": str(checkpoint_path),
        "qat_best_checkpoint": str(output_path),
        "qat_last_checkpoint": str(last_path),
        "family": family,
        "rank_ratio": rank_ratio,
        "max_epochs": args.epochs,
        "qat_mode": args.qat_mode,
        "qat_modules": qat_module_count(student),
        "source_pre_recovery_top1_accuracy": source_pre_recovery_acc,
        "source_best_top1_accuracy": source_metrics.get("best_top1_accuracy", ""),
        "qat_pre_top1_accuracy": pre_metrics["top1_accuracy"],
        "qat_best_top1_accuracy": best_acc,
        "qat_final_top1_accuracy": final_acc,
        "qat_best_full_top1_accuracy": full_best_acc,
        "qat_final_full_top1_accuracy": full_final_acc,
        "fp32_model_size_mb": source_metrics.get(
            "model_size_mb", measure_model_size_mb(student)
        ),
        "qat_model_size_mb": measure_model_size_mb(student),
        "estimated_int8_model_size_mb": estimated_int8_model_size_mb(student),
        "qat_best_checkpoint_mb": checkpoint_size_mb(output_path),
        "qat_last_checkpoint_mb": checkpoint_size_mb(last_path),
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.disable_observer_after < 0:
        raise ValueError("disable-observer-after must be non-negative.")
    if not 0.0 <= args.hard_target_weight <= 1.0:
        raise ValueError("hard-target-weight must be in [0, 1].")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    print(f"[*] Max epochs per model: {args.epochs}")
    print(f"[*] Checkpoints: {len(args.checkpoints)}")

    trainloader, testloader = data_loaders(args.batch_size, args.num_workers)
    teacher = build_teacher(args.teacher_checkpoint, device)
    manifest_rows = []
    for checkpoint in args.checkpoints:
        manifest_rows.append(
            train_one_checkpoint(
                checkpoint, teacher, trainloader, testloader, device, args
            )
        )
        write_csv(
            args.output_dir / "manifest.csv",
            manifest_rows,
            [
                "source_checkpoint",
                "qat_best_checkpoint",
                "qat_last_checkpoint",
                "family",
                "rank_ratio",
                "max_epochs",
                "qat_mode",
                "qat_modules",
                "source_pre_recovery_top1_accuracy",
                "source_best_top1_accuracy",
                "qat_pre_top1_accuracy",
                "qat_best_top1_accuracy",
                "qat_final_top1_accuracy",
                "qat_best_full_top1_accuracy",
                "qat_final_full_top1_accuracy",
                "fp32_model_size_mb",
                "qat_model_size_mb",
                "estimated_int8_model_size_mb",
                "qat_best_checkpoint_mb",
                "qat_last_checkpoint_mb",
            ],
        )

    print(f"[*] QAT complete. Manifest: {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
