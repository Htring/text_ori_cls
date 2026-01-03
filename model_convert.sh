model_dir="PP-LCNet_x1_0_doc_ori_infer"
onnx_output_dir="PP-LCNet_x1_0_doc_ori_infer"

# model_dir="PP-LCNet_x1_0_textline_ori_infer"
# onnx_output_dir="PP-LCNet_x1_0_textline_ori_infer"

paddlex \
    --paddle2onnx \
    --paddle_model_dir $model_dir \
    --onnx_model_dir $onnx_output_dir \
    --opset_version 7  # 指定要使用的 ONNX opset 版本