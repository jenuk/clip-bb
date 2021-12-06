import os

import PIL
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import clip

import config2object

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    def __init__(self, dataset, interpolation="bicubic"):
        self.dataset = config2object.instantiate_from_config(dataset)
        self.extra_keys = dataset.extra_keys

        # hardcode clip transformation
        self.size = 224 # size used by clip
        self.clip_transform = transforms.Compose([
            # resize, crop and to rgb is done manually here
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        self.interpolation = {"linear":PIL.Image.LINEAR,
                              "bilinear":PIL.Image.BILINEAR,
                              "bicubic":PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def size_to_splits(self, size):
        raise NotImplemented("Inherit from this class and overwrite this method")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, caption, extra = self.dataset[i]

        # clip tokens
        tokens = clip.tokenize(batch["caption"], truncate=True)

        # preprocess images
        if not image.mode == "RGB":
            image = image.convert("RGB")

        size = image.size

        # transform image to self.size while keeping aspect ratio
        new_size = ((size[0] * self.size)//min(size), (size[1] * self.size)//min(size))
        image = image.resize(new_size, resample=self.interpolation)

        starts = self.size_to_splits(new_size)

        slides = []
        for x, y in starts:
            cropped = image.crop((x, y, x+self.size, y+self.size))
            slides.append(self.clip_transform(cropped))
        slides = torch.stack(slides_clip)

        factor = min(size)/self.size # go back from resized coordinates to original
        starts = (torch.tensor(starts)*factor).int()

        return slides, starts, tokens, extra



class SlideDataset(BaseDataset):
    def __init__(self, stride=8, max_ratio=2, **kwargs):
        super(BaseDataset, self).__init__(**kwargs)
        self.stride = stride
        self.max_ratio = max_ratio


    def size_to_splits(self, size):
        if size[0] > self.size:
            offset = max(size[0]//2 - (self.max_ratio-1)*self.size, 0)
            end = min(size[0] - self.size, size[0]//2 + (self.max_ratio-1)*self.size)

            starts = [[delta, 0] for delta in range(offset, end, self.slide_step_size)]
        elif size[1] > self.size:
            offset = max(size[1]//2 - (self.max_ratio-1)*self.size, 0)
            end = min(size[1] - self.size, size[1]//2 + (self.max_ratio-1)*self.size)

            starts = [[0, delta] for delta in range(offset, end, self.slide_step_size)]
        else:
            starts = [[0, 0]]

        return starts



class CenterDataset(BaseDataset):
    def __init__(self, max_ratio=2, **kwargs):
        super(BaseDataset, self).__init__(**kwargs)
        self.max_ratio = max_ratio


    def size_to_splits(self, size):
        if size[0] > self.size:
            offset = max(size[0]//2 - (self.max_ratio-1)*self.size, 0)
            end = min(size[0] - self.size, size[0]//2 + (self.max_ratio-1)*self.size)

            starts = [[(offset+end)/2, 0]]
        elif size[1] > self.size:
            offset = max(size[1]//2 - (self.max_ratio-1)*self.size, 0)
            end = min(size[1] - self.size, size[1]//2 + (self.max_ratio-1)*self.size)

            starts = [[0, (offset+end)/2]]
        else:
            starts = [[0, 0]]

        return starts
