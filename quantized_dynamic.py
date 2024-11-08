import onnx
from onnxconverter_common import float16


def convert_to_fp16(input_path, output_path):
    print(f"Starting FP16 conversion of {input_path}")

    try:
        # 加载模型
        model = onnx.load(input_path)

        # 转换为FP16，但保持某些操作为float32
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            op_block_list=['Resize', 'Reshape', 'Concat', 'Slice']  # 添加不转换的操作
        )

        # 保存模型
        onnx.save(model_fp16, output_path)

        print(f"Conversion completed. Model saved to {output_path}")

    except Exception as e:
        print(f"Error during conversion: {str(e)}")


def safe_fp16_convert(input_path, output_path):
    model = onnx.load(input_path)

    # 列出所有不支持fp16的操作
    blocked_ops = [
        'Resize', 'Reshape', 'Concat', 'Slice',
        'Shape', 'Gather', 'NonMaxSuppression',
        'TopK', 'Range', 'Cast', 'Clip', 'Pad'
    ]

    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=blocked_ops,
    )

    onnx.save(model_fp16, output_path)


# 使用示例
input_model = "/home/yjam/PycharmProjects/shenNeng/weights/fireAndSmoke_yolon_ghost_p2_ema.onnx"
output_model = "/home/yjam/PycharmProjects/shenNeng/weights/fireAndSmoke_yolon_ghost_p2_ema_fp16.onnx"

safe_fp16_convert(input_model, output_model)
