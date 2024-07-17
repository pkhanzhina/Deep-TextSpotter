from PIL import Image
import numpy as np


class VerticalResize:
    def __init__(self, h):
        self.h = h

    def __call__(self, img):
        w, h = img.size
        new_w = int(self.h * w / h)
        return img.resize((new_w, self.h))


class HorizontalResize:
    def __init__(self, w):
        self.w = w

    def __call__(self, img):
        w, h = img.size
        if w > self.w:
            return img.resize((self.w, h))
        else:
            return img


class RandomHorizontalCrop:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img):
        w, h = img.size
        low, high = self.scale[0], min(self.scale[1], w)
        left_w = np.random.randint(low, high)
        right_w = w - np.random.randint(low, high)
        crop_area = (left_w, 0, right_w, h)
        return img.crop(crop_area)


class RandomHorizontalResize:
    def __init__(self, scale=(0.8, 1.2)):
        self.scale = scale

    def __call__(self, img):
        w, h = img.size
        low, high = int(self.scale[0] * w), int(self.scale[1] * w)
        new_w = np.random.randint(low, high)
        return img.resize((new_w, h))


class GaussianNoise:
    def __init__(self, mean=0, std=0.02):
        self.mean, self.std = mean, std

    def __call__(self, img):
        noise = np.random.normal(size=img.size[::-1])
        noise = self.std * noise + self.mean
        return (img + noise).astype(np.float32)


class ImageWhitening:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = (np.asarray(img) - self.mean) / self.std
        return img
