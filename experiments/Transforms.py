import os

import numpy as np
import matplotlib.pyplot as plt

from monai.transforms import MapTransform, LoadPNG, Resize


class PNGsToStack(MapTransform):
    def __init__(self, keys, spatial_size=[256, 256, 24]):
        super().__init__(keys)
        self.png_loader = LoadPNG(image_only=True)
        self.resizer_xy = Resize(spatial_size[:-1])
        self.resizer_z = Resize(spatial_size)

    def __call__(self, data):
        for key in self.keys:
            stack = []
            for png_file in os.listdir(data[key]):
                png_file = self.png_loader(os.path.join(data[key], png_file))
                # Image just duplicated along channels to produce grayscale PNG
                png_file = png_file[..., 0]
                png_file = np.expand_dims(png_file, axis=0)
                png_file = self.resizer_xy(png_file)
                stack.append(png_file.squeeze())
            stack = np.stack(stack, axis=-1)
            stack = np.expand_dims(stack, axis=0)
            stack = self.resizer_z(stack)
            data[key] = stack
        return data


class Normalize(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            data[key] = (data[key] - data[key].mean()) / data[key].std()
        return data
