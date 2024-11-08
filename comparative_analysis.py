import onnx
import os
from onnxsim import simplify
import onnxruntime as ort
import numpy as np
import time
import psutil


def compare_models(original_path, simplified_path):
    """
    模型大小和结构对比
    :param original_path:
    :param simplified_path:
    :return:
    """
    # 加载模型
    original_model = onnx.load(original_path)
    simplified_model = onnx.load(simplified_path)

    # 比较文件大小
    original_size = os.path.getsize(original_path) / (1024 * 1024)  # MB
    simplified_size = os.path.getsize(simplified_path) / (1024 * 1024)  # MB

    print(f"原始模型大小: {original_size:.2f} MB")
    print(f"简化模型大小: {simplified_size:.2f} MB")
    print(f"大小减少: {(original_size - simplified_size) / original_size * 100:.2f}%")

    # 比较节点数量
    original_nodes = len(original_model.graph.node)
    simplified_nodes = len(simplified_model.graph.node)

    print(f"\n原始模型节点数: {original_nodes}")
    print(f"简化模型节点数: {simplified_nodes}")
    print(f"节点减少: {(original_nodes - simplified_nodes) / original_nodes * 100:.2f}%")


def benchmark_inference_speed(model_path, input_shape, num_iterations=100, use_gpu=True):
    # 指定运行设备
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

    # 创建推理会话
    session = ort.InferenceSession(
        model_path,
        providers=providers
    )

    # 打印当前使用的设备
    print(f"Running on: {session.get_providers()}")

    # 准备输入数据
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # GPU数据处理
    if use_gpu:
        # 对于ONNX Runtime，直接使用numpy数组即可
        # 它会自动处理GPU内存分配
        pass

    # 预热
    print("Warming up...")
    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    # 测速
    print(f"Running {num_iterations} iterations...")
    start_time = time.time()
    for _ in range(num_iterations):
        session.run(None, {input_name: dummy_input})
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations * 1000  # 转换为ms
    return avg_time


def compare_speed(original_path, simplified_path, input_shape, use_gpu=True):
    print(f"Testing with input shape: {input_shape}")
    print(f"Using GPU: {use_gpu}")

    original_time = benchmark_inference_speed(original_path, input_shape, use_gpu=use_gpu)
    simplified_time = benchmark_inference_speed(simplified_path, input_shape, use_gpu=use_gpu)

    print(f"\nOriginal model average inference time: {original_time:.2f} ms")
    print(f"Simplified model average inference time: {simplified_time:.2f} ms")
    print(f"Speed improvement: {(original_time - simplified_time) / original_time * 100:.2f}%")


def measure_memory_usage(model_path, input_shape):
    process = psutil.Process(os.getpid())
    base_memory = process.memory_info().rss / (1024 * 1024)  # MB

    # 指定运行设备
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    # 创建推理会话
    session = ort.InferenceSession(
        model_path,
        providers=providers
    )

    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # 运行推理
    session.run(None, {input_name: dummy_input})

    peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
    return peak_memory - base_memory


def compare_memory(original_path, simplified_path, input_shape):
    original_memory = measure_memory_usage(original_path, input_shape)
    simplified_memory = measure_memory_usage(simplified_path, input_shape)

    print(f"\n原始模型内存使用: {original_memory:.2f} MB")
    print(f"简化模型内存使用: {simplified_memory:.2f} MB")
    print(f"内存减少: {(original_memory - simplified_memory) / original_memory * 100:.2f}%")


def comprehensive_comparison(original_path, simplified_path, input_shape):
    print("=" * 50)
    print("开始全面对比分析")
    print("=" * 50)

    # 1. 模型大小和结构对比
    print("\n1. 模型大小和结构对比")
    compare_models(original_path, simplified_path)

    # 2. 推理速度对比
    print("\n2. 推理速度对比")
    compare_speed(original_path, simplified_path, input_shape)

    # 3. 内存使用对比
    print("\n3. 内存使用对比")
    compare_memory(original_path, simplified_path, input_shape)

    print("\n" + "=" * 50)


input_shape = (1, 3, 640, 640)  # 根据你的模型修改
comprehensive_comparison("/home/yjam/PycharmProjects/shenNeng/weights/FireAndSmoke_yolon_aifi.onnx", "/home/yjam/PycharmProjects/shenNeng/weights/FireAndSmoke_yolon_aifi_simplify.onnx", input_shape)

