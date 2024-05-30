from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


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
        image = Image.open(img_path).convert('RGB')
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