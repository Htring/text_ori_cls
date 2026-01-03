from typing import List, Tuple
import numpy as np
from PIL import Image, ImageOps

def exif_transpose(img: Image.Image) -> Image.Image:
    try:
        img_corrected = ImageOps.exif_transpose(img)
        if img_corrected is None:
            return img
        return img_corrected
    except Exception as e:
        return img

def img_to_ndarray(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB')
    return np.array(img)

def load_img2np(img_path):
    img = Image.open(img_path)
    img = exif_transpose(img)
    return img_to_ndarray(img)


class ClsPostProcess:
    def __init__(self, label_list: List[str]):
        self.label_list = label_list

    def __call__(self, preds: np.ndarray) -> List[Tuple[str, float]]:
        pred_idxs = preds.argmax(axis=1)
        decode_out = [
            (self.label_list[int(idx)], preds[i, int(idx)])
            for i, idx in enumerate(pred_idxs)
        ]
        return decode_out