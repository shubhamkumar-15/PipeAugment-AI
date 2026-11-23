from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

REQUIRED_COLUMNS = [
    "image_id", "dataset", "relative_path", "label",
    "class_index", "split", "group_id", "sha256"
]


class PipeAugmentDataset(Dataset):
    """
    Manifest-based PyTorch Dataset for PipeAugment AI.
    Loads images directly from manifest rows, strictly preserving EXIF orientation
    and converting to RGB.
    """

    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        transform: Optional[Callable] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Manifest missing required columns: {missing_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_full_path = self.dataset_root / row["relative_path"]

        if not img_full_path.exists():
            cand = PROJECT_ROOT / row["relative_path"]
            if cand.exists():
                img_full_path = cand
            else:
                cand2 = (self.dataset_root / ".." / ".." / row["relative_path"]).resolve()
                if cand2.exists():
                    img_full_path = cand2
                else:
                    raise FileNotFoundError(f"Image file does not exist: {img_full_path}")

        with Image.open(img_full_path) as img:
            # Correct EXIF orientation and ensure 3-channel RGB
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            if self.transform is not None:
                tensor_img = self.transform(img)
            else:
                tensor_img = T.ToTensor()(img)

        label_idx = int(row["class_index"])
        meta = {
            "image_id": str(row["image_id"]),
            "relative_path": str(row["relative_path"]),
            "label": str(row["label"]),
            "group_id": str(row["group_id"]),
            "sha256": str(row["sha256"]),
        }
        return tensor_img, label_idx, meta


def get_baseline_transforms(weights: Any = None) -> Callable:
    """
    Returns deterministic evaluation-style preprocessing based on pretrained weights.
    Applies exact ImageNet resize (256), center crop (224), and standard normalization.
    NO stochastic augmentations (no random flips, no jitter, no random crops) for Baseline A.
    """
    if weights is not None and hasattr(weights, "transforms"):
        return weights.transforms()

    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
