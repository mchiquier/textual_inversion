import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random
import glob

imagenet_templates_smallest = [
    'a photo of a {}',
]

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]

imagenet_dual_templates_small = [
    'a photo of a {} with {}',
    'a rendering of a {} with {}',
    'a cropped photo of the {} with {}',
    'the photo of a {} with {}',
    'a photo of a clean {} with {}',
    'a photo of a dirty {} with {}',
    'a dark photo of the {} with {}',
    'a photo of my {} with {}',
    'a photo of the cool {} with {}',
    'a close-up photo of a {} with {}',
    'a bright photo of the {} with {}',
    'a cropped photo of a {} with {}',
    'a photo of the {} with {}',
    'a good photo of the {} with {}',
    'a photo of one {} with {}',
    'a close-up photo of the {} with {}',
    'a rendition of the {} with {}',
    'a photo of the clean {} with {}',
    'a rendition of a {} with {}',
    'a photo of a nice {} with {}',
    'a good photo of a {} with {}',
    'a photo of the nice {} with {}',
    'a photo of the small {} with {}',
    'a photo of the weird {} with {}',
    'a photo of the large {} with {}',
    'a photo of a cool {} with {}',
    'a photo of a small {} with {}',
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]
# per_img_token_list = ['b','c','d','e','f','g','h','i','j']


class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 edit_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 crop_p=0.0,
                 procedural_task='ab',
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 ):

        self.data_root = data_root
        self.edit_root = edit_root

        # self.image_paths = [os.path.join(self.data_root, file_path)
        #                     for file_path in os.listdir(self.data_root)]
        # self.image_paths_edited = [os.path.join(self.edit_root, file_path)
        #                            for file_path in os.listdir(self.data_root)]
        self.image_paths = sorted(glob.glob(os.path.join(self.data_root, '*.*')))
        self.image_paths_edited = sorted(glob.glob(os.path.join(self.edit_root, '*.*')))

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self.num_images_edited = len(self.image_paths_edited)
        assert self.num_images == self.num_images_edited, \
            (f"Number of images in data_root ({self.num_images}) and "
             f"edit_root ({self.num_images_edited}) must match.")
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), \
                (f"Can't use per-image tokens when the training set "
                 f"contains more than {len(per_img_token_list)} tokens. "
                 f"To enable larger sets, add more tokens to 'per_img_token_list'.")

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        self.flip_p = flip_p
        self.crop_p = crop_p
        self.procedural_task = procedural_task

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}

        image = Image.open(self.image_paths[i % self.num_images])
        image_edited = Image.open(self.image_paths_edited[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")
            image_edited = image_edited.convert("RGB")

        # TODO: what exactly is happening here?
        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
            text = random.choice(imagenet_dual_templates_small).format(
                placeholder_string, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(placeholder_string)

        # NOTE: caption seems to be overwritten in the trainnig loop anyway..
        example["caption"] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        img_edited = np.array(image_edited).astype(np.uint8)

        if self.center_crop:
            # crop = min(img.shape[0], img.shape[1])
            # h, w, = img.shape[0], img.shape[1]
            # img = img[(h - crop) // 2:(h + crop) // 2,
            #     (w - crop) // 2:(w + crop) // 2]

            img = transforms.functional.center_crop(image, min(image.size))
            img_edited = transforms.functional.center_crop(image_edited, min(image_edited.size))

        image = Image.fromarray(img)
        image_edited = Image.fromarray(img_edited)
        if self.size is not None:
            image = image.resize((self.size, self.size),
                                 resample=self.interpolation)
            image_edited = image_edited.resize((self.size, self.size),
                                               resample=self.interpolation)

        # image = self.flip(image)
        # image_edited = self.flip(image_edited)
        if np.random.rand() < self.flip_p:
            image = transforms.functional.hflip(image)
            image_edited = transforms.functional.hflip(image_edited)
        image = np.array(image).astype(np.uint8)
        image_edited = np.array(image_edited).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["edited"] = (image_edited / 127.5 - 1.0).astype(np.float32)
        return example
