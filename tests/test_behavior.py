import torch
import pytest
from torch.utils.cpp_extension import load
import os

# 自动通过 PyTorch JIT 编译加载 custom_op.cpp
_cpp_loaded = False
try:
    # 假设 custom_op.cpp 在 /workspace/custom_op.cpp
    # 如果找不到，可以根据你的实际路径调整
    op_path = "/workspace/custom_op.cpp"
    if os.path.exists(op_path):
        custom_ops = load(
            name="custom_ops_ext",
            sources=[op_path],
            verbose=True,
            is_python_module=False
        )
        _cpp_loaded = True
except Exception as e:
    print(f"Failed to JIT compile custom op: {e}")

# 确保加载了编译好的自定义算子动态库
try:
    torch.ops.load_library("/workspace/build/libcustom_op.so")
except Exception:
    try:
        torch.ops.load_library("libcustom_op.so")
    except Exception:
        pass

def test_standard_contiguous_execution() -> None:
    x = torch.randn(4, 4, dtype=torch.float32)
    w = torch.randn(4, dtype=torch.float32)
    
    # 调用自定义算子和参考实现
    out = torch.ops.custom_ops.custom_weighted_sum(x, w)
    ref = torch.matmul(x, w) # 或者你定义的数学参考逻辑
    
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

def test_non_contiguous_tensor_strides() -> None:
    # 构造非连续张量（通过切片或转置）
    x_base = torch.randn(8, 8, dtype=torch.float32)
    x_sliced = x_base[:, ::2] # 列方向步长为 2，非连续
    w = torch.randn(4, dtype=torch.float32)
    
    assert not x_sliced.is_contiguous(), "Test tensor must be non-contiguous"
    
    # 真实调用自定义算子
    out = torch.ops.custom_ops.custom_weighted_sum(x_sliced, w)
    
    # 高精度参考实现（显式转成连续或直接用标准运算对比）
    ref = torch.matmul(x_sliced.contiguous(), w) 
    
    # 如果 C++ 代码没有处理 strides，这里算出的结果会与 ref 不一致，从而触发失败！
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

def test_torch_compile_layout_integration() -> None:
    if hasattr(torch, "compile"):
        x_base = torch.randn(8, 8, dtype=torch.float32)
        x_sliced = x_base[:, ::2]
        w = torch.randn(4, dtype=torch.float32)
        
        # 测试 torch.compile 是否会因为 stride 假设错误导致崩溃或结果不一致
        compiled_op = torch.compile(lambda inp, weights: torch.ops.custom_ops.custom_weighted_sum(inp, weights))
        
        out = compiled_op(x_sliced, w)
        ref = torch.matmul(x_sliced.contiguous(), w)
        
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
