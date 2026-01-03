from classifier import TextPicOriCLS
from utils import load_img2np
import os


if __name__ == """__main__""":

    model = "./PP-LCNet_x1_0_doc_ori_infer/inference.onnx"
    param = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "label_list": ["0", "90", "180", "270"],
        "use_cuda": False,
        "image_shape": [3, 224, 224]
    }
    # model = "PP-LCNet_x1_0_textline_ori_infer/inference.onnx"
    # param = {
    #     "mean": [0.485, 0.456, 0.406],
    #     "std": [0.229, 0.224, 0.225],
    #     "label_list": ["0_degree", "180_degree"],
    #     "use_cuda": True,
    #     "image_shape": [3, 80, 160]
    # }
    classifier = TextPicOriCLS(model, **param)
    images = []
    file_paths = []
    for file in os.listdir("./test_pngs"):
        file_path = os.path.join("./test_pngs", file)
        images.append(load_img2np(file_path))
        file_paths.append(file_path)
    res = classifier(images)
    for u, pre in zip(file_paths, res[0]):
        print(u, pre)