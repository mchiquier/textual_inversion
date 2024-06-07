import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class AFHQ(Dataset):
    def __init__(self, root_dir: Path, split: str, transform=None):
        self.root_dir = root_dir
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.cat_dir = self.split_dir / "cat"
        self.dog_dir = self.split_dir / "dog"

        self.cat_images = list(self.cat_dir.rglob("*.jpg"))
        self.dog_images = list(self.cat_dir.rglob("*.jpg"))

        self.image_list = self.cat_images + self.dog_images
        self.labels = [0] * len(self.cat_images) + [1] * len(self.dog_images)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class Cats(AFHQ):
    def __init__(self, root_dir: Path, split: str, transform=None):
        super().__init__(root_dir, split, transform)
        self.image_list = self.cat_images
        self.labels = [0] * len(self.cat_images)


class Dogs(AFHQ):
    def __init__(self, root_dir: Path, split: str, transform=None):
        super().__init__(root_dir, split, transform)
        self.image_list = self.dog_images
        self.labels = [1] * len(self.dog_images)


class ImageEdits(Dataset):
    def __init__(self, images_dir: Path, edits_dir: Path, transform=None) -> None:
        self.images_dir = images_dir
        self.edits_dir = edits_dir
        self.transform = transform

        self.image_list = list(self.images_dir.rglob("*"))

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def convert_to_np(img_path: Path):
        image = Image.open(img_path).convert("RGB")
        return np.array(image).transpose(2, 0, 1)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img_edit_path = self.edits_dir / img_path.name

        image = self.convert_to_np(img_path)
        image_edit = self.convert_to_np(img_edit_path)

        images = np.concatenate([image, image_edit])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1

        if self.transform:
            images = self.transform(images)

        image, image_edit = images.chunk(2)
        return image, image_edit


class TextualInversionEdits(ImageEdits):
    def __init__(
        self,
        images_dir: Path,
        edits_dir: Path,
        placeholder_str: list[str],
        transform=None,
    ) -> None:
        super().__init__(images_dir, edits_dir, transform)
        self.placeholder_str = placeholder_str

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img_edit_path = self.edits_dir / img_path.name

        image = self.convert_to_np(img_path)
        image_edit = self.convert_to_np(img_edit_path)

        images = np.concatenate([image, image_edit])
        images = torch.tensor(images)

        if self.transform:
            images = self.transform(images)

        images = images / 255.0
        image, image_edit = images.chunk(2)
        # TODO: support multiple placeholder strings
        prompt = self.placeholder_str[0]
        return image, image_edit, prompt


class TextualInversionEval(TextualInversionEdits):
    def __init__(
        self,
        images_dir: Path,
        placeholder_str: list[str],
        edits_dir: Path = None,
        transform=None,
    ) -> None:
        super().__init__(images_dir, edits_dir, placeholder_str, transform)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = self.convert_to_np(img_path)
        image = torch.tensor(image)

        if self.transform:
            image = self.transform(image)

        image = image / 255.0
        # TODO: support multiple placeholder strings
        prompt = self.placeholder_str[0]
        return image, prompt
