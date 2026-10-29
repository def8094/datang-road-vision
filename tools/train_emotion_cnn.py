#!/usr/bin/env python3
"""Train the document-style FER CNN with PyTorch and OpenCV preprocessing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


class EmotionDataset(Dataset):
    def __init__(self, root: Path, split: str, augment: bool = False, image_size: int = 48, channels: int = 1):
        self.samples = []
        self.augment = augment
        self.image_size = image_size
        self.channels = channels
        split_root = root / split
        classes = sorted(path.name for path in split_root.iterdir() if path.is_dir())
        self.class_to_id = {name: index for index, name in enumerate(classes)}
        for class_name in classes:
            self.samples.extend((path, self.class_to_id[class_name]) for path in sorted((split_root / class_name).iterdir()) if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if not self.samples:
            raise FileNotFoundError(f"No emotion images found under {split_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR if self.channels == 3 else cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Unable to read {path}")
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if self.channels == 1:
            image = cv2.equalizeHist(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        if self.augment and np.random.random() < 0.5:
            image = cv2.flip(image, 1)
        if self.augment and np.random.random() < 0.35:
            center = self.image_size / 2.0
            angle = float(np.random.uniform(-8.0, 8.0))
            matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (self.image_size, self.image_size), borderMode=cv2.BORDER_REFLECT_101)
        if self.augment and np.random.random() < 0.3:
            alpha = float(np.random.uniform(0.85, 1.15))
            beta = float(np.random.uniform(-12.0, 12.0))
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        tensor = torch.from_numpy(image).float().div(255.0)
        if self.channels == 1:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.permute(2, 0, 1)
            tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor, label


class SimpleEmotionCNN(nn.Module):
    """A compact residual CNN using the document's OpenCV + PyTorch route."""

    def __init__(self, classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            self._block(32, 32), nn.MaxPool2d(2), nn.Dropout2d(0.10),
            self._block(32, 64), nn.MaxPool2d(2), nn.Dropout2d(0.12),
            self._block(64, 128), nn.MaxPool2d(2), nn.Dropout2d(0.15),
            self._block(128, 192), nn.MaxPool2d(2), nn.Dropout2d(0.18),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(192, 128), nn.SiLU(inplace=True),
            nn.Dropout(0.35), nn.Linear(128, classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Module:
        shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        return _ResidualBlock(in_channels, out_channels, shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, shortcut: nn.Module):
        super().__init__()
        self.shortcut = shortcut
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.shortcut(x))


class ResNetEmotionCNN(nn.Module):
    """ImageNet-initialized CNN head for the higher-quality RAF-DB images."""

    def __init__(self, classes: int, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(512, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(arch: str, classes: int, pretrained: bool = False) -> nn.Module:
    if arch == "resnet18":
        return ResNetEmotionCNN(classes, pretrained=pretrained)
    if arch == "simple":
        return SimpleEmotionCNN(classes)
    raise ValueError(f"Unsupported emotion architecture: {arch}")


def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum = correct = total = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--arch", choices=("simple", "resnet18"), default="simple")
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=1)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--time-hours", type=float, default=1.8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("training/driver_emotion/weights/best.pt"))
    args = parser.parse_args()
    train_set = EmotionDataset(args.data, "train", augment=True, image_size=args.image_size, channels=args.channels)
    val_set = EmotionDataset(args.data, args.val_split, image_size=args.image_size, channels=args.channels)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
    device = torch.device(args.device)
    model = build_model(args.arch, len(train_set.class_to_id), pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.3)
    class_counts = np.bincount([label for _, label in train_set.samples], minlength=len(train_set.class_to_id))
    weights = torch.tensor(class_counts.sum() / np.maximum(class_counts, 1), dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy, stale = -1.0, 0
    deadline = time.monotonic() + args.time_hours * 3600
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            if time.monotonic() >= deadline:
                break
            images, labels = images.to(device), torch.as_tensor(labels, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
        val_loss, accuracy = evaluate(model, val_loader, device, criterion)
        scheduler.step(accuracy)
        print(f"epoch={epoch:03d} train_loss={loss_sum / max(len(train_loader), 1):.4f} val_loss={val_loss:.4f} val_accuracy={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy, stale = accuracy, 0
            torch.save({
                "model": model.state_dict(),
                "classes": list(train_set.class_to_id),
                "epoch": epoch,
                "accuracy": accuracy,
                "arch": args.arch,
                "image_size": args.image_size,
                "channels": args.channels,
            }, args.output)
        else:
            stale += 1
        if stale >= args.patience or time.monotonic() >= deadline:
            break
    metrics = {
        "best_accuracy": best_accuracy,
        "classes": list(train_set.class_to_id),
        "checkpoint": str(args.output),
        "arch": args.arch,
        "image_size": args.image_size,
        "channels": args.channels,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
