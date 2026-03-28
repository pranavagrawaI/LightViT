# scripts/04_evaluate.py
import torch
import torchvision
import torchvision.transforms as transforms
from models.baseline_vit import LightViTBaseline
from core.metrics import count_parameters, measure_latency_ms, measure_model_size_mb


@torch.inference_mode()
def run_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Auditing Baseline on: {device}")

    # 1. Prepare Clean Test Data (No Augmentations)
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )

    # 2. Load the Checkpoint
    model = LightViTBaseline().to(device)
    checkpoint_path = "checkpoints/baseline_fp32.pth"

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Check if the weights are wrapped in a nested dictionary
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"[!] Error: {checkpoint_path} not found. Did the training save a file?")
        return

    # 3. Measure Accuracy
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total

    # 4. Measure Physical Reality
    print("\n" + "=" * 40)
    print("      OFFICIAL BASELINE AUDIT")
    print("=" * 40)
    print(f"Top-1 Accuracy:      {accuracy:.2f}%")

    params = count_parameters(model)
    print(f"Total Parameters:    {params:,}")

    size_mb = measure_model_size_mb(model)
    print(f"Model Size (VRAM):   {size_mb:.2f} MB")

    # Measure latency with a single sample
    sample = torch.randn(1, 3, 32, 32).to(device)
    latency = measure_latency_ms(model, sample)
    print(f"Inference Latency:   {latency:.4f} ms")
    print("=" * 40)

    # Calculate the "Death Floor"
    print(f"\n[*] TARGET FOR LIGHTVIT:")
    print(f"Minimum Accuracy Required (> -7%): {accuracy - 7.0:.2f}%")
    print(f"Target Parameters (< 1.5M):       SUCCESS if < 1,500,000")


if __name__ == "__main__":
    run_audit()
