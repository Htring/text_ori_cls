from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_all_providers,
    get_device,
)
from loguru import logger
from typing import Union, List
from pathlib import Path
import os
import traceback
import numpy as np


class OrtInferSession(object):
    def is_cuda_available(self) -> bool:
        device = get_device()
        if device == "GPU":
            if "CUDAExecutionProvider" in self.exist_providers:
                return True
            else:
                logger.warning(
                    f"CUDAExecutionProvider is not in available providers ({self.exist_providers}). Use {self.default_provider} inference by default."
                )
                return False
        else:
            logger.warning("not gpu onnx")
            return False

    @staticmethod
    def _verify_model(model_path: Union[str, Path, None]):
        if model_path is None:
            raise ValueError("model_path is None!")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")

    def __init__(self, model_path: str, use_cuda: bool = False):
        self._verify_model(model_path)
        self.exist_providers = get_all_providers()
        self.default_provider = self.exist_providers[0]
        default_used_providers = [
            ("CPUExecutionProvider", dict(arena_extend_strategy="kSameAsRequested"))
        ]
        if use_cuda and self.is_cuda_available():
            default_used_providers.insert(
                0,
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),  # type: ignore
            )
        sess_opt = self._init_sess_opts()
        self.session = InferenceSession(
            str(model_path),
            sess_options=sess_opt,
            providers=default_used_providers,
        )

    @staticmethod
    def _init_sess_opts() -> SessionOptions:
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = True
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_nums = os.cpu_count()
        intra_op_num_threads = -1
        if intra_op_num_threads != -1 and 1 <= intra_op_num_threads <= cpu_nums:
            sess_opt.intra_op_num_threads = intra_op_num_threads
        inter_op_num_threads = -1
        if inter_op_num_threads != -1 and 1 <= inter_op_num_threads <= cpu_nums:
            sess_opt.inter_op_num_threads = inter_op_num_threads
        return sess_opt

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), [input_content]))
        try:
            return self.session.run(self.get_output_names(), input_dict)[0]
        except Exception as e:
            error_info = traceback.format_exc()
            raise Exception(error_info) from e

    def get_input_names(self) -> List[str]:
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character") -> List[str]:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in meta_dict.keys():
            return True
        return False
