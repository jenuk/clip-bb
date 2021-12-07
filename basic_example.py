import os
import pandas
import PIL
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BasicDataset:
    def __init__(self, csv_file, data_root):
        self.data_paths = csv_file
        self.data_root = os.path.expanduser(data_root)
        self.data_csv = pandas.read_csv(csv_file, sep=",")

        self._length = len(self.data_csv["path"])
        self.labels = {
            "relative_file_path_": [l for l in self.data_csv["path"]],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.data_csv["path"]],
            "caption": [cap for cap in self.data_csv["caption"]],
        }

        self.extra_keys = {"relative_file_path_", "caption", "width", "height"}


    def __len__(self):
        return self._length


    def __getitem__(self, i):
        extra = dict()
        for key in self.extra_keys & self.labels.keys():
            extra[key] = self.labels[key][i]

        image = Image.open(self.labels["file_path_"][i])

        extra["width"] = image.size[0]
        extra["height"] = image.size[1]

        # after using clip-bb use the new start_x and start_y values:
        # size = min(image.size)
        # image = image.crop(start_x, start_y, start_x+size, start_y+size)
        # use your other transforms here

        return image, extra["caption"], extra
