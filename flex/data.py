from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class UltrasoundCsvDataset(Dataset):
    """CSV dataset with image, mask, and integer class label columns."""

    def __init__(
        self,
        csv_path: str | Path,
        root: str | Path | None = None,
        image_size: int = 224,
        image_col: str = "image",
        mask_col: str = "mask",
        label_col: str = "label",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root = Path(root) if root is not None else self.csv_path.parent
        self.image_size = int(image_size)
        self.image_col = image_col
        self.mask_col = mask_col
        self.label_col = label_col
        with self.csv_path.open(newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"No rows found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self._resolve(row[self.image_col])).convert("RGB")
        mask = Image.open(self._resolve(row[self.mask_col])).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        image_tensor = TF.to_tensor(image)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return image_tensor, mask_tensor, label, row[self.image_col]
