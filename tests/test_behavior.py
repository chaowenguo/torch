import torch
import pytest
import os
from torch.utils.cpp_extension import load

try:
    op_path = "/workspace/custom_op.cpp"
    if os.path.exists(op_path):
        load(
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
    features = torch.randn(16, 8, dtype=torch.float32)
    indices = torch.tensor([0, 3, 5, 7], dtype=torch.int64)
    scales = torch.tensor([1.0, 0.5, 2.0, 1.5], dtype=torch.float32)
    
    out = torch.ops.custom_ops.sparse_gather_scale(features, indices, scales)
    ref = features[indices] * scales.unsqueeze(1)
    
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

def test_non_contiguous_features_strides() -> None:
    base = torch.randn(16, 16, dtype=torch.float32)
    features = base[:, ::2] # non-contiguous slice (hidden_dim = 8, col stride = 2)
    indices = torch.tensor([1, 2, 4], dtype=torch.int64)
    scales = torch.tensor([0.8, 1.2, 1.0], dtype=torch.float32)
    
    assert not features.is_contiguous()
    
    out = torch.ops.custom_ops.sparse_gather_scale(features, indices, scales)
    ref = features[indices] * scales.unsqueeze(1)
    
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

def test_torch_compile_fullgraph_integration() -> None:
    if hasattr(torch, "compile"):
        base = torch.randn(16, 16, dtype=torch.float32)
        features = base[:, ::2]
        indices = torch.tensor([0, 2], dtype=torch.int64)
        scales = torch.tensor([1.0, 1.0], dtype=torch.float32)
        
        compiled_op = torch.compile(
            lambda f, idx, sc: torch.ops.custom_ops.sparse_gather_scale(f, idx, sc),
            fullgraph=True
        )
        
        out = compiled_op(features, indices, scales)
        ref = features[indices] * scales.unsqueeze(1)
        
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
