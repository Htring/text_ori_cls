import math
from typing import Any, Union, List
import numpy as np
import cv2
import time
from engine import OrtInferSession
from utils import ClsPostProcess, load_img2np


class TextPicOriCLS(object):
    def __init__(
        self,
        model_path: str,
        mean,
        std,
        image_shape: List[float],
        label_list: List[str],
        use_cuda: bool = True,
        batch_size: int = 6,
    ) -> None:
        self.cls_image_shape = image_shape
        self.mean = np.array(mean).reshape(3, 1, 1)
        self.std = np.array(std).reshape(3, 1, 1)
        self.cls_batch_num = batch_size
        self.post_progress = ClsPostProcess(label_list)

        self.engine = OrtInferSession(model_path, use_cuda)

    def __call__(self, img_list: Union[np.ndarray, List[np.ndarray]]) -> Any:
        start_time = time.perf_counter()
        if isinstance(img_list, np.ndarray):
            img_list = [img_list]
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        img_num = len(img_list)
        cls_res = [("", 0.0)] * img_num
        batch_num = self.cls_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)
            prob_out = self.engine(norm_img_batch)
            cls_result = self.post_progress(prob_out)
            for rno, (label, score) in enumerate(cls_result):
                cls_res[indices[beg_img_no + rno]] = (label, score)
        elapse = time.perf_counter() - start_time
        return cls_res, elapse

    def resize_norm_img(self, img: np.ndarray) -> np.ndarray:
        img_c, img_h, img_w = self.cls_image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(img_h * ratio) > img_w:
            resized_w = img_w
        else:
            resized_w = int(math.ceil(img_h * ratio))
        resized_image = cv2.resize(img, (resized_w, img_h))
        resized_image = resized_image.astype("float32")
        if img_c == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        # 标准化
        resized_image = (resized_image - self.mean) / self.std
        padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        return padding_im

