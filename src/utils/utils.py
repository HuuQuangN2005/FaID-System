import torch
import torch.nn as nn 
from torchvision.transforms import functional as TF

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = [hp, vp, max_wh - w - hp, max_wh - h - vp]
        
        # Dùng torchvision.transforms.functional.pad
        return TF.pad(image, padding, fill=0, padding_mode='constant')