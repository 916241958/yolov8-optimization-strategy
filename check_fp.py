import torch
import numpy as np


def check_gpu_detailed_capabilities():
    if not torch.cuda.is_available():
        print("No CUDA device available!")
        return

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n{'=' * 50}")
        print(f"GPU {i}: {props.name}")
        print(f"{'=' * 50}")

        # 基本信息
        print("\n1. 基本信息:")
        print(f"- 计算能力: {props.major}.{props.minor}")
        print(f"- 总内存: {props.total_memory / 1024 ** 2:.1f} MB")
        print(f"- 多处理器数量: {props.multi_processor_count}")

        # 精度支持
        print("\n2. 精度支持:")
        print("基础精度:")
        print(f"- FP16 (半精度): {props.major >= 6}")
        print(f"- FP32 (单精度): True")
        print(f"- FP64 (双精度): {props.major >= 6}")
        print(f"- INT8: {props.major >= 6}")
        print(f"- INT4: {props.major >= 7}")
        print(f"- BF16: {props.major >= 8}")

        # 内存信息
        print("\n3. 内存信息:")
        print(f"- 总显存: {props.total_memory / 1024 ** 2:.1f} MB")

        # CUDA功能
        print("\n4. CUDA功能:")
        print(f"- CUDA版本: {torch.version.cuda}")
        print(f"- cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"- cuDNN启用: {torch.backends.cudnn.enabled}")
        print(f"- cuDNN确定性模式: {torch.backends.cudnn.deterministic}")
        print(f"- cuDNN基准测试模式: {torch.backends.cudnn.benchmark}")

        # 当前GPU状态
        print("\n5. 当前GPU状态:")
        print(f"- 当前设备内存分配:")
        print(f"  已用: {torch.cuda.memory_allocated(i) / 1024 ** 2:.1f} MB")
        print(f"  缓存: {torch.cuda.memory_reserved(i) / 1024 ** 2:.1f} MB")

        # 架构特性
        print("\n6. 架构特性:")
        arch_dict = {
            1: "Tesla",
            2: "Fermi",
            3: "Kepler",
            5: "Maxwell",
            6: "Pascal",
            7: "Volta/Turing",
            8: "Ampere",
            9: "Hopper"
        }
        arch_name = arch_dict.get(props.major, "Unknown")
        print(f"- GPU架构: {arch_name}")
        print(f"- 计算能力版本: {props.major}.{props.minor}")

        # 推荐的优化策略
        print("\n7. 推荐的优化策略:")
        if props.major < 6:
            print("- 建议使用FP32进行计算")
            print("- 避免使用低精度计算，因为硬件不支持")
            print("- Maxwell架构优化建议:")
            print("  * 使用cuDNN进行深度学习加速")
            print("  * 优化内存访问模式")
            print("  * 使用异步内存传输")
        else:
            print("可用的优化策略:")
            if props.major >= 8:
                print("- 可使用BF16进行训练")
            if props.major >= 7:
                print("- 可使用INT8/INT4量化")
            if props.major >= 6:
                print("- 可使用FP16混合精度训练")

        if props.total_memory < 4 * 1024 ** 3:
            print("\n内存优化建议:")
            print("- 建议使用较小的batch size")
            print("- 考虑使用梯度累积")
            print("- 可以考虑使用量化减少内存使用")
            print("- 对于MX110的具体建议:")
            print("  * batch size建议不超过16")
            print("  * 考虑使用模型剪枝")
            print("  * 使用特征图压缩")

        print("\n" + "=" * 50)


# 运行检查
check_gpu_detailed_capabilities()
