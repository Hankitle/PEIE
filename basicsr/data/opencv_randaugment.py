import cv2
import numpy as np
import random
from .degradation import add_JPEG_noise, add_Gaussian_noise

def Contrast(img, v):
    assert v >= 0.0
    mean = np.mean(img, axis=(0, 1), keepdims=True) + 0.5
    return np.clip(mean * (1 - v) + img * v, 0.0, 1.0)

def Posterize(img, v):
    v = max(1, int(v))
    shift = 8 - v 
    img_quantized = np.floor(img * 255).astype(np.uint8)
    img_quantized = np.clip((img_quantized >> shift) << shift, 0, 255) 
    return img_quantized.astype(np.float32) / 255.0

def Sharpness(img, v):
    assert v >= 0.0
    kernel = np.array([[1, 1, 1],
                       [1, 5, 1],
                       [1, 1, 1]], dtype=np.float32)  / 13
    sharp = cv2.filter2D(img, -1, kernel)
    return np.clip(sharp * (1 - v) + img * v, 0.0, 1.0) 

class OpenCVAugment:
    def __init__(self, n=2):
        self.n = n
        self.augment_list = [
            (Contrast, 0.3, 0.8),
            (Posterize, 6, 8),
            (Sharpness, 0.05, 0.8),
            (add_JPEG_noise),
            (add_Gaussian_noise),
        ]
        self.weights = [1.5, 1, 2, 5, 0.5]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n, weights=self.weights)
        ops = set(ops)
        for op in ops:
            if callable(op):
                img = op(img)
            else:
                op, min_val, max_val = op
                val = min_val + float(max_val - min_val) * random.random()
                img = op(img, val)
        return img
