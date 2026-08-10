# Task title: Fix Stride-Ignored Memory Access and Fake Tensor Layout Mismatch in Custom Weighted-Sum Operator

A custom C++ extension providing a weighted-sum operator (`custom_weighted_sum`) has been integrated into the repository. While the operator functions correctly on standard, contiguous inputs, it produces silent numerical errors and incorrect output values when provided with non-contiguous tensors—such as sliced, permuted, or transposed views. 

Furthermore, when executed under `torch.compile`, Inductor plans incorrect memory layouts and strides because the registered fake/meta tensor implementation incorrectly assumes a contiguous output layout regardless of the input strides.

### Requirements

1. **Stride-Aware Kernel Execution**: The underlying C++ kernel must correctly compute memory offsets using the input tensor's actual `strides` rather than assuming flat, contiguous memory storage. Sliced or transposed inputs must yield precise mathematical results matching an analytical reference instead of silently returning corrupted or shifted data.
2. **Meta / Fake Tensor Alignment**: The registration for `torch.compile` (meta kernel) must correctly preserve and propagate input strides to the output tensor. It must not force a default contiguous layout if non-contiguous or transposed tensors are supplied.
3. **Behavioral Invariants**: 
   - Calling the operator on non-contiguous views must produce identical results to explicitly calling `.contiguous()` followed by the operation, but without requiring unnecessary memory allocation or copying.
   - The operator must compose cleanly with `torch.compile` without triggering layout mismatches or incorrect stride optimization plans in Inductor.

