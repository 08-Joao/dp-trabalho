from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


CLASS_NAMES = {
    0: "Ignorar / sem dado",
    1: "Fundo",
    2: "Construcao",
    3: "Estrada",
    4: "Agua",
    5: "Solo exposto",
    6: "Floresta / vegetacao",
    7: "Agricultura",
}

CLASS_COLORS = np.array(
    [
        [0, 0, 0],
        [180, 180, 180],
        [220, 40, 40],
        [245, 190, 30],
        [40, 120, 240],
        [170, 110, 60],
        [35, 150, 70],
        [185, 220, 70],
    ],
    dtype=np.uint8,
)


@dataclass
class TrainConfig:
    data_root: str = "data/LoveDA"
    image_size: int = 256
    train_count: int = 100
    val_count: int = 30
    test_count: int = 30
    batch_size: int = 8
    epochs: int = 5
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 2
    num_classes: int = 8
    ignore_index: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def find_loveda_pairs(data_root: str | Path) -> list[tuple[Path, Path]]:
    root = Path(data_root)
    image_dirs = sorted(p for p in root.rglob("images_png") if p.is_dir())
    pairs: list[tuple[Path, Path]] = []
    for image_dir in image_dirs:
        mask_dir = image_dir.parent / "masks_png"
        if not mask_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / image_path.name
            if mask_path.exists():
                pairs.append((image_path, mask_path))
    if not pairs:
        raise FileNotFoundError(
            f"Nenhum par imagem/mascara encontrado em {root}. "
            "Esperado: subpastas images_png e masks_png."
        )
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]], train_count: int, val_count: int, test_count: int, seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    needed = train_count + val_count + test_count
    if len(shuffled) < needed:
        raise ValueError(f"A base tem {len(shuffled)} pares, mas foram pedidos {needed}.")
    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count : needed]
    return train, val, test


def remap_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.int64)
    mask[mask == 255] = 0
    valid = mask[mask != 0]
    # LoveDA comum: 0 fundo, 1 construcao, 2 estrada, ..., 6 agricultura.
    # Enunciado: 0 ignorar, 1 fundo, 2 construcao, ..., 7 agricultura.
    if mask.max(initial=0) <= 6 and valid.size:
        mask = mask + 1
    mask[(mask < 0) | (mask > 7)] = 0
    return mask.astype(np.int64)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask = np.clip(mask, 0, len(CLASS_COLORS) - 1)
    return CLASS_COLORS[mask]


def present_classes(mask: np.ndarray) -> list[str]:
    ids = sorted(int(x) for x in np.unique(mask) if int(x) in CLASS_NAMES)
    return [f"{idx} - {CLASS_NAMES[idx]}" for idx in ids]


def denormalize_image(image: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.transpose(1, 2, 0)
    return np.clip((image * std + mean) * 255, 0, 255).astype(np.uint8)


def build_transforms(image_size: int, augment: bool) -> A.Compose:
    steps: list[A.BasicTransform] = []
    if augment:
        steps.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.10,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.7,
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(0.75, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.6,
                ),
            ]
        )
    steps.extend(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return A.Compose(steps)


class LoveDADataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], image_size: int, augment: bool = False):
        self.pairs = pairs
        self.transforms = build_transforms(image_size, augment)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = remap_mask(np.array(Image.open(mask_path)))
        transformed = self.transforms(image=image, mask=mask)
        image = transformed["image"].transpose(2, 0, 1).astype(np.float32)
        mask = transformed["mask"].astype(np.int64)
        return torch.from_numpy(image), torch.from_numpy(mask)


def make_loaders(
    train_pairs: list[tuple[Path, Path]],
    val_pairs: list[tuple[Path, Path]],
    test_pairs: list[tuple[Path, Path]],
    cfg: TrainConfig,
    augment: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = LoveDADataset(train_pairs, cfg.image_size, augment=augment)
    val_ds = LoveDADataset(val_pairs, cfg.image_size, augment=False)
    test_ds = LoveDADataset(test_pairs, cfg.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader, test_loader


def build_model(name: str, num_classes: int = 8) -> nn.Module:
    if name == "unet_scratch":
        return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=num_classes)
    if name in {"unet_tl", "unet_tl_aug"}:
        return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=num_classes)
    raise ValueError(f"Modelo desconhecido: {name}")


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)
    keep = target != ignore_index
    pred = pred[keep]
    target = target[keep]
    idx = target * num_classes + pred
    return torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(cm: torch.Tensor, ignore_index: int = 0) -> dict[str, float]:
    cm = cm.float()
    tp = torch.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    union = tp + fp + fn
    iou = torch.where(union > 0, tp / union.clamp_min(1), torch.nan)
    pixel_acc = tp.sum() / cm.sum().clamp_min(1)
    valid = torch.ones_like(iou, dtype=torch.bool)
    valid[ignore_index] = False
    return {
        "pixel_accuracy": float(pixel_acc.cpu()),
        "mean_iou": float(torch.nanmean(iou[valid]).cpu()),
        "iou_building": float(iou[2].cpu()),
        "iou_road": float(iou[3].cpu()),
        "iou_vegetation": float(iou[6].cpu()),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, cfg: TrainConfig) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    cm = torch.zeros(cfg.num_classes, cfg.num_classes, dtype=torch.long, device=cfg.device)
    for images, masks in loader:
        images = images.to(cfg.device)
        masks = masks.to(cfg.device)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        cm += confusion_matrix(preds, masks, cfg.num_classes, cfg.ignore_index).to(cfg.device)
    metrics = metrics_from_confusion(cm, cfg.ignore_index)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def train_one_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    output_dir: str | Path = "outputs",
) -> tuple[nn.Module, pd.DataFrame, dict[str, float]]:
    model = build_model(model_name, cfg.num_classes).to(cfg.device)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rows = []
    best_iou = -1.0
    best_path = Path(output_dir) / f"{model_name}_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"{model_name} epoca {epoch}/{cfg.epochs}"):
            images = images.to(cfg.device)
            masks = masks.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= max(len(train_loader.dataset), 1)
        val_metrics = evaluate(model, val_loader, criterion, cfg)
        row = {"model": model_name, "epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        rows.append(row)
        if val_metrics["mean_iou"] > best_iou:
            best_iou = val_metrics["mean_iou"]
            torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=cfg.device))
    history = pd.DataFrame(rows)
    return model, history, {"best_val_mean_iou": best_iou, "checkpoint": str(best_path)}


def run_experiments(
    cfg: TrainConfig,
    output_dir: str | Path = "outputs",
    experiments: Iterable[tuple[str, bool]] | None = None,
) -> tuple[dict[str, nn.Module], pd.DataFrame, pd.DataFrame, tuple[list, list, list]]:
    seed_everything(cfg.seed)
    pairs = find_loveda_pairs(cfg.data_root)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, cfg.train_count, cfg.val_count, cfg.test_count, cfg.seed)
    if experiments is None:
        experiments = [
            ("unet_scratch", False),
            ("unet_tl", False),
            ("unet_tl_aug", True),
        ]
    models = {}
    histories = []
    test_rows = []
    for model_name, augment in experiments:
        train_loader, val_loader, test_loader = make_loaders(train_pairs, val_pairs, test_pairs, cfg, augment=augment)
        model, history, info = train_one_model(model_name, train_loader, val_loader, cfg, output_dir)
        criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
        test_metrics = evaluate(model, test_loader, criterion, cfg)
        models[model_name] = model
        histories.append(history)
        test_rows.append({"model": model_name, **test_metrics, **info})
    history_df = pd.concat(histories, ignore_index=True)
    metrics_df = pd.DataFrame(test_rows)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    history_df.to_csv(Path(output_dir) / "training_history.csv", index=False)
    metrics_df.to_csv(Path(output_dir) / "test_metrics.csv", index=False)
    (Path(output_dir) / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return models, history_df, metrics_df, (train_pairs, val_pairs, test_pairs)


def show_dataset_examples(pairs: list[tuple[Path, Path]], n: int = 5, figsize: tuple[int, int] = (11, 4)) -> None:
    sample = pairs[:n]
    for image_path, mask_path in sample:
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = remap_mask(np.array(Image.open(mask_path)))
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(image)
        axes[0].set_title("Imagem original")
        axes[0].axis("off")
        axes[1].imshow(mask_to_rgb(mask))
        axes[1].set_title("Mascara real")
        axes[1].axis("off")
        fig.suptitle(f"{image_path.name}\nClasses: {', '.join(present_classes(mask))}", fontsize=10)
        plt.tight_layout()
        plt.show()


@torch.no_grad()
def show_predictions(
    model: nn.Module,
    test_pairs: list[tuple[Path, Path]],
    cfg: TrainConfig,
    n: int = 5,
    alpha: float = 0.45,
) -> None:
    dataset = LoveDADataset(test_pairs[:n], cfg.image_size, augment=False)
    model.eval()
    model.to(cfg.device)
    for idx in range(len(dataset)):
        image_t, mask_t = dataset[idx]
        logits = model(image_t.unsqueeze(0).to(cfg.device))
        pred = logits.argmax(1).squeeze(0).cpu().numpy()
        image = denormalize_image(image_t.numpy())
        mask = mask_t.numpy()
        pred_rgb = mask_to_rgb(pred)
        overlay = np.clip((1 - alpha) * image + alpha * pred_rgb, 0, 255).astype(np.uint8)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ["Imagem original", "Mascara real", "Mascara prevista", "Sobreposicao"]
        for ax, arr, title in zip(axes, [image, mask_to_rgb(mask), pred_rgb, overlay], titles):
            ax.imshow(arr)
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.show()


def plot_history(history_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model_name, group in history_df.groupby("model"):
        axes[0].plot(group["epoch"], group["train_loss"], marker="o", label=f"{model_name} treino")
        axes[0].plot(group["epoch"], group["val_loss"], marker="o", linestyle="--", label=f"{model_name} val")
        axes[1].plot(group["epoch"], group["val_mean_iou"], marker="o", label=model_name)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[1].set_title("IoU medio de validacao")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("mIoU")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()
