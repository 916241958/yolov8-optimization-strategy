# 简化ONNX模型，去除冗余操作，融合算子
import onnx
from onnxsim import simplify

# 加载模型
model = onnx.load("/home/yjam/PycharmProjects/shenNeng/weights/FireAndSmoke_yolon_aifi.onnx")

# 简化模型
model_simplified, check = simplify(model)

# 检查并保存
if check:
    onnx.save(model_simplified, "/home/yjam/PycharmProjects/shenNeng/weights/FireAndSmoke_yolon_aifi_simplify.onnx")
    print("Successfully simplified")
