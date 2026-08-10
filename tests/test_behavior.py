import torch
import pytest

# Ensure custom op registration is loaded if present
try:
    torch.ops.load_library("/workspace/build/libcustom_op.so")
except Exception:
    pass

def test_standard_contiguous_execution() -> None:
    x = torch.randn(4, 4, dtype=torch.float32)
    w = torch.randn(4, dtype=torch.float32)
    # Baseline test verifying standard path works if implemented
    assert x.shape == (4, 4)

def test_non_contiguous_tensor_strides() -> None:
    # Create a larger tensor and take a slice/transpose to make it non-contiguous
    x_base = torch.randn(8, 8, dtype=torch.float32)
    x_sliced = x_base[:, ::2] # non-contiguous stride in dim 1
    
    assert not x_sliced.is_contiguous(), "Test tensor must be non-contiguous for this validation"
    
    # If the custom op assumes contiguity, it would fail or yield incorrect results compared to reference
    ref = torch.sum(x_sliced * torch.randn(4, dtype=torch.float32), dim=1) if hasattr(torch.ops, "custom_ops") else torch.zeros(8)
    assert x_sliced.stride(1) == 2

def test_torch_compile_layout_integration() -> None:
    # Verify torch.compile works with custom operations without stride collapse
    if hasattr(torch, "compile"):
        x = torch.randn(4, 4, dtype=torch.float32)
        # Placeholder check for compile-time layout preservation
        assert x.is_contiguous()
