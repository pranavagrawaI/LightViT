"""Refine the baseline checkpoint with stronger regularization and optional teacher distillation."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from timm.data.mixup import Mixup
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.resnet import ResNet
from torchvision.models.vision_transformer import VisionTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_vit import LightViTBaseline  # noqa: E402


CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 2))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--mixup-alpha", type=float, default=0.8)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument(
        "--teacher", choices=("none", "resnet50", "vit_b_16"), default="none"
    )
    parser.add_argument("--distill-weight", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "baseline_fp32.pth",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "baseline_refine_last.pth",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data_loaders(
    batch_size: int, num_workers: int
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
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


def load_student_checkpoint(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)


def forward_student_features(
    model: LightViTBaseline, images: torch.Tensor
) -> torch.Tensor:
    batch_size = images.shape[0]
    tokens = model.patch_embed(images)
    cls_tokens = model.cls_token.expand(batch_size, -1, -1)
    tokens = torch.cat((cls_tokens, tokens), dim=1)
    tokens = tokens + model.pos_embed
    tokens = model.blocks(tokens)
    return model.norm(tokens[:, 0])


class FeatureDistiller(nn.Module):
    def __init__(self, teacher_name: str, student_dim: int):
        super().__init__()
        self.teacher_name = teacher_name
        self.teacher, teacher_dim = self._build_teacher(teacher_name)
        self.projector = nn.Sequential(
            nn.LayerNorm(student_dim), nn.Linear(student_dim, teacher_dim)
        )

        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

    def _build_teacher(
        self, teacher_name: str
    ) -> tuple[ResNet | VisionTransformer, int]:
        if teacher_name == "resnet50":
            teacher = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT
            )
            teacher.eval()
            teacher_fc = cast(nn.Linear, teacher.fc)
            return teacher, teacher_fc.in_features

        if teacher_name == "vit_b_16":
            teacher = torchvision.models.vit_b_16(
                weights=torchvision.models.ViT_B_16_Weights.DEFAULT
            )
            teacher.eval()
            teacher_head = cast(nn.Linear, teacher.heads.head)
            teacher_dim = teacher_head.in_features
            return teacher, teacher_dim

        raise ValueError(f"Unsupported teacher: {teacher_name}")

    def _student_to_teacher_inputs(self, images: torch.Tensor) -> torch.Tensor:
        cifar_mean = torch.tensor(
            CIFAR100_MEAN, device=images.device, dtype=images.dtype
        ).view(1, 3, 1, 1)
        cifar_std = torch.tensor(
            CIFAR100_STD, device=images.device, dtype=images.dtype
        ).view(1, 3, 1, 1)
        imagenet_mean = torch.tensor(
            IMAGENET_MEAN, device=images.device, dtype=images.dtype
        ).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(
            IMAGENET_STD, device=images.device, dtype=images.dtype
        ).view(1, 3, 1, 1)

        images = images * cifar_std + cifar_mean
        images = F.interpolate(
            images, size=(224, 224), mode="bilinear", align_corners=False
        )
        return (images - imagenet_mean) / imagenet_std

    def _teacher_features(self, images: torch.Tensor) -> torch.Tensor:
        inputs = self._student_to_teacher_inputs(images)

        if self.teacher_name == "resnet50":
            teacher = cast(ResNet, self.teacher)
            x = teacher.conv1(inputs)
            x = teacher.bn1(x)
            x = teacher.relu(x)
            x = teacher.maxpool(x)
            x = teacher.layer1(x)
            x = teacher.layer2(x)
            x = teacher.layer3(x)
            x = teacher.layer4(x)
            x = teacher.avgpool(x)
            return torch.flatten(x, 1)

        teacher = cast(VisionTransformer, self.teacher)
        x = teacher._process_input(inputs)
        class_token = teacher.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = teacher.encoder(x)
        return x[:, 0]

    @torch.no_grad()
    def teacher_features(self, images: torch.Tensor) -> torch.Tensor:
        return self._teacher_features(images)

    def loss(
        self, student_features: torch.Tensor, images: torch.Tensor
    ) -> torch.Tensor:
        teacher_features = F.normalize(self.teacher_features(images).float(), dim=-1)
        projected_student = F.normalize(
            self.projector(student_features).float(), dim=-1
        )
        return 1.0 - F.cosine_similarity(
            projected_student, teacher_features, dim=-1
        ).mean()


def evaluate(
    model: LightViTBaseline, loader: torch.utils.data.DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        progress = tqdm(loader, desc="Validation", leave=False)
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(images)

            predictions = logits.argmax(dim=1)
            total += targets.size(0)
            correct += predictions.eq(targets).sum().item()
            progress.set_postfix(acc=f"{100.0 * correct / total:.2f}%")

    return 100.0 * correct / total


def save_best_checkpoint(
    model: LightViTBaseline,
    checkpoint_path: Path,
    best_acc: float,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "refinement": {
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "mixup_alpha": args.mixup_alpha,
                "cutmix_alpha": args.cutmix_alpha,
                "mixup_prob": args.mixup_prob,
                "teacher": args.teacher,
                "distill_weight": args.distill_weight,
                "scheduler": "cosine_annealing",
            },
        },
        checkpoint_path,
    )


def save_resume_checkpoint(
    model: LightViTBaseline,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    resume_path: Path,
    epoch: int,
    best_acc: float,
    distiller: FeatureDistiller | None,
) -> None:
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }
    if distiller is not None:
        payload["projector"] = distiller.projector.state_dict()

    torch.save(payload, resume_path)


def maybe_load_resume_checkpoint(
    model: LightViTBaseline,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    resume_path: Path,
    distiller: FeatureDistiller | None,
) -> tuple[int, float]:
    if not resume_path.exists():
        return 0, 0.0

    checkpoint = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    try:
        scheduler.load_state_dict(checkpoint["scheduler"])
    except Exception as exc:
        print(f"[!] Skipping scheduler state from resume checkpoint: {exc}")
    scaler.load_state_dict(checkpoint["scaler"])

    if distiller is not None and "projector" in checkpoint:
        distiller.projector.load_state_dict(checkpoint["projector"])

    return checkpoint.get("epoch", 0), checkpoint.get("best_acc", 0.0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Refining on device: {device}")
    print(f"[*] Baseline checkpoint: {args.baseline_checkpoint}")

    trainloader, testloader = get_data_loaders(args.batch_size, args.num_workers)

    model = LightViTBaseline().to(device)
    load_student_checkpoint(model, args.baseline_checkpoint, device)
    print("[*] Loaded baseline weights")

    distiller: FeatureDistiller | None = None
    if args.teacher != "none":
        student_head = cast(nn.Linear, model.head)
        distiller = FeatureDistiller(
            args.teacher, student_dim=student_head.in_features
        ).to(device)
        distiller.teacher.eval()
        print(f"[*] Teacher distillation enabled: {args.teacher}")
    else:
        print("[*] Teacher distillation disabled")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    soft_target_criterion = SoftTargetCrossEntropy()
    mixup_enabled = args.mixup_prob > 0.0 and (
        args.mixup_alpha > 0.0 or args.cutmix_alpha > 0.0
    )
    mixup_fn = (
        Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode="batch",
            label_smoothing=args.label_smoothing,
            num_classes=100,
        )
        if mixup_enabled
        else None
    )
    if mixup_enabled:
        print(
            "[*] Mixup/CutMix enabled: "
            f"mixup_alpha={args.mixup_alpha}, cutmix_alpha={args.cutmix_alpha}"
        )
    else:
        print("[*] Mixup/CutMix disabled")

    optimization_params: list[dict[str, object]] = [{"params": model.parameters()}]
    if distiller is not None:
        optimization_params.append({"params": distiller.projector.parameters()})

    optimizer = optim.AdamW(
        optimization_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    start_epoch = 0
    best_acc = evaluate(model, testloader, device)
    print(f"[*] Starting validation accuracy: {best_acc:.2f}%")

    if args.resume:
        start_epoch, best_acc = maybe_load_resume_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            args.resume_checkpoint,
            distiller,
        )
        print(
            f"[*] Resumed from epoch {start_epoch} with best accuracy {best_acc:.2f}%"
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start_lr = optimizer.param_groups[0]["lr"]
        running_total = 0.0
        running_ce = 0.0
        running_distill = 0.0
        progress = tqdm(
            trainloader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]", leave=False
        )

        for step, (images, targets) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mixed_targets = targets
            if mixup_fn is not None:
                images, mixed_targets = mixup_fn(images, targets)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=device.type == "cuda"):
                student_features = forward_student_features(model, images)
                logits = model.head(student_features)
                if mixup_fn is not None:
                    ce_loss = soft_target_criterion(logits, mixed_targets)
                else:
                    ce_loss = criterion(logits, targets)

                if distiller is not None:
                    distill_loss = distiller.loss(student_features, images)
                    loss = (
                        1.0 - args.distill_weight
                    ) * ce_loss + args.distill_weight * distill_loss
                else:
                    distill_loss = torch.zeros((), device=device)
                    loss = ce_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if distiller is not None:
                torch.nn.utils.clip_grad_norm_(
                    distiller.projector.parameters(), max_norm=1.0
                )
            scaler.step(optimizer)
            scaler.update()

            running_total += loss.item()
            running_ce += ce_loss.item()
            running_distill += distill_loss.item()
            progress.set_postfix(
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                loss=f"{running_total / step:.3f}",
                ce=f"{running_ce / step:.3f}",
                kd=f"{running_distill / step:.4f}",
            )

        scheduler.step()
        epoch_end_lr = optimizer.param_groups[0]["lr"]

        acc = evaluate(model, testloader, device)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"LR: {epoch_start_lr:.2e} -> {epoch_end_lr:.2e} | "
            f"Train Loss: {running_total / len(trainloader):.3f} | "
            f"Train CE: {running_ce / len(trainloader):.3f} | "
            f"Train KD: {running_distill / len(trainloader):.4f} | "
            f"Val Acc: {acc:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            save_best_checkpoint(
                model, args.baseline_checkpoint, best_acc, epoch + 1, args
            )
            print(
                f"[*] New best refined baseline saved to {args.baseline_checkpoint} ({best_acc:.2f}%)"
            )

        save_resume_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            args.resume_checkpoint,
            epoch + 1,
            best_acc,
            distiller,
        )

    print(f"[*] Refinement complete. Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
