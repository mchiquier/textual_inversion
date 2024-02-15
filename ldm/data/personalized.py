import os
import numpy as np
import PIL
import glob
import pdb
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

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
                 set="train",
                 repeats=100,
                 interpolation="bicubic",
                 center_crop=False,
                 size=None,
                 horizontal_flip=False,
                 random_crop=False,
                 gaussian_blur=False,
                 gaussian_noise=False,
                 flip_p=0.5,
                 crop_p=0.8,
                 blur_p=0.4,
                 noise_p=0.4,
                 placeholder_token="*",
                 per_image_tokens=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 ):

        self.data_root = data_root
        self.edit_root = edit_root

        self.horizontal_flip = horizontal_flip
        self.random_crop = random_crop
        self.gaussian_blur = gaussian_blur
        self.gaussian_noise = gaussian_noise

        self.image_paths = sorted(glob.glob(os.path.join(self.data_root, '*.*')))
        self.image_paths_edited = sorted(glob.glob(os.path.join(self.edit_root, '*.*')))

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self.num_images_edited = len(self.image_paths_edited)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text
        self.set = set
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
        self.blur_p = blur_p
        self.noise_p = noise_p

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

        # First center crop.
        if self.center_crop:
            img = transforms.functional.center_crop(
                image, min(image.size))
            img_edited = transforms.functional.center_crop(
                image_edited, min(image_edited.size))

        # Then random crop if applicable.
        if 'train' in self.set.lower():
            # random cropping
            if self.random_crop and np.random.rand() < self.crop_p:
                # Crop to random square that is 80% of the smallest image dimension.
                width, height = image.size
                crop_size = int(min(width, height) * 0.8)

                # Randomly choose top left corner.
                i = np.random.randint(0, height - crop_size)
                j = np.random.randint(0, width - crop_size)

                # Apply the same crop to both images
                image = transforms.functional.crop(
                    image, i, j, crop_size, crop_size)
                image_edited = transforms.functional.crop(
                    image_edited, i, j, crop_size, crop_size)

        # Then resize to final model size.
        image = Image.fromarray(img)
        image_edited = Image.fromarray(img_edited)
        if self.size is not None:
            image = image.resize((self.size, self.size),
                                 resample=self.interpolation)
            image_edited = image_edited.resize((self.size, self.size),
                                               resample=self.interpolation)

        # Finally apply all other random data augmentations if applicable.
        if 'train' in self.set.lower():
            # random horizontal flip
            if self.horizontal_flip and np.random.rand() < self.flip_p:
                image = transforms.functional.hflip(image)
                image_edited = transforms.functional.hflip(image_edited)

            # gaussian blurring
            if self.gaussian_blur and np.random.rand() < self.blur_p:
                sigma = torch.rand(1) * (2 - 0.1) + 0.1
                # gausian_blur_transformation_13 = T.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2))
                image = F.gaussian_blur(image, (3, 7), [sigma, sigma])
                image_edited = F.gaussian_blur(image_edited, (3, 7), [sigma, sigma])

                # image = gausian_blur_transformation_13(image)
                # image_edited = gausian_blur_transformation_13(image_edited)

            # add gaussian noise
            if self.gaussian_noise and np.random.rand() < self.noise_p:
                image_np = np.array(image)
                image_edited_np = np.array(image_edited)

                noise = np.random.normal(0, 1, image_np.shape)
                noisy_image_np = image_np + noise
                noisy_image_edited_np = image_edited_np + noise

                noisy_image_np = np.clip(noisy_image_np, 0, 255)
                noisy_image_edited_np = np.clip(noisy_image_edited_np, 0, 255)

                image = Image.fromarray(noisy_image_np.astype(np.uint8))
                image_edited = Image.fromarray(noisy_image_edited_np.astype(np.uint8))

        image = np.array(image).astype(np.uint8)
        image_edited = np.array(image_edited).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["edited"] = (image_edited / 127.5 - 1.0).astype(np.float32)

        return example
