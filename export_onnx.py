import onnx
import onnxruntime


model = onnx.load('/home/yjam/PycharmProjects/shenneng/weights/fireAndSmoke_yolon_ghost_p2_ema.onnx')
onnx.checker.check_model(model)  # 验证Onnx模型是否准确

for input_node in model.graph.input:
    # print(input_node.type.tensor_type.shape)
    input_node.type.tensor_type.shape.dim[2].dim_value = 320
    input_node.type.tensor_type.shape.dim[3].dim_value = 320
    print(input_node.type.tensor_type.shape)

onnx.checker.check_model(model)
onnx.save(model, '/home/yjam/PycharmProjects/shenneng/weights/fireAndSmoke_yolon_ghost_p2_ema.onnx')
