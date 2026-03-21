# scripts/01_train_baseline.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.baseline_vit import LightViTBaseline
from core.metrics import count_parameters, measure_latency_ms, measure_model_size_mb
import os


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Forging on device: {device}")

    # 1. Aggressive Data Augmentation (Mandatory for ViTs)
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )

    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )

    # 2. Instantiate and Measure Ground Truth
    model = LightViTBaseline().to(device)
    print("\n--- Baseline Metrics Before Training ---")
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params}")
    sample = torch.randn(1, 3, 32, 32).to(device)
    latency = measure_latency_ms(model, sample)
    print(f"Latency: {latency:.2f} ms")
    print("----------------------------------------\n")
    model_size_mb = measure_model_size_mb(model)
    print(f"Model Size: {model_size_mb:.2f} MB")

    # 3. The ViT Optimization Recipe
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)  # 100 Epochs

    # 4. The Training Loop
    epochs = 100
    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping stabilizes ViT training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.0 * correct / total
        print(
            f"Epoch [{epoch + 1}/{epochs}] | Loss: {running_loss / len(trainloader):.3f} | Acc: {acc:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "checkpoints/baseline_fp32.pth")
            print(f"[*] New Best Baseline Saved! ({best_acc:.2f}%)")


if __name__ == "__main__":
    main()
