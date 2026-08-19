import torch
import pytest
import os
from torch.utils.cpp_extension import load

try:
    op_path = "/workspace/custom_op.cpp"
    if os.path.exists(op_path):
        custom_ops = load(
            name="custom_ops_ext",
            sources=[op_path],
            verbose=True,
            is_python_module=False
        )
except Exception as e:
    print(f"Failed to JIT compile custom op: {e}")

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
    
    out = torch.ops.custom_ops.custom_weighted_sum(x, w)
    ref = torch.matmul(x, w)
    
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

def test_non_contiguous_tensor_strides() -> None:
    x_base = torch.randn(8, 8, dtype=torch.float32)
    x_sliced = x_base[:, ::2]
    w = torch.randn(4, dtype=torch.float32)
    
    assert not x_sliced.is_contiguous(), "Test tensor must be non-contiguous"
    
    out = torch.ops.custom_ops.custom_weighted_sum(x_sliced, w)
    ref = torch.matmul(x_sliced.contiguous(), w) 
    
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

def test_torch_compile_fullgraph_integration() -> None:
    if hasattr(torch, "compile"):
        x_base = torch.randn(8, 8, dtype=torch.float32)
        x_sliced = x_base[:, ::2]
        w = torch.randn(4, dtype=torch.float32)
        
        # Enforce fullgraph=True to guarantee that fake/meta registration is used and no fallback/graph breaks occur
        compiled_op = torch.compile(
            lambda inp, weights: torch.ops.custom_ops.custom_weighted_sum(inp, weights),
            fullgraph=True
        )
        
        out = compiled_op(x_sliced, w)
        ref = torch.matmul(x_sliced.contiguous(), w)
        
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
