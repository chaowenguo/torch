# Fix Stride-Ignored Index Slicing and Missing Meta Registration in Custom Sparse Gather-Scale Operator

A custom C++ PyTorch extension providing a sparse embedding gather and scaling operator (`sparse_gather_scale`) has been added to the repository. While the operator works on purely contiguous feature matrices, it yields silent data corruption and incorrect values when supplied with non-contiguous or sliced weight/feature views (e.g., transposed embedding tables or sliced features).

Additionally, execution under `torch.compile(..., fullgraph=True)` fails completely because the operator lacks a registered meta/fake tensor implementation, causing Inductor graph breaks or runtime unsupported operator errors.

### Requirements

1. **Stride-Aware C++ Execution**: The underlying CPU kernel must compute memory addresses using the actual tensor strides (`strides()`) rather than assuming flat, contiguous memory storage. Sliced or transposed feature views must be indexed correctly.
2. **Meta / Fake Tensor Registration**: Implement and register a corresponding meta/fake kernel (`sparse_gather_scale_meta`) that correctly validates input dimensions/indices and returns an output tensor of the expected shape and options without executing real data kernels.
3. **Behavioral Invariants**: 
   - Calling the operator on non-contiguous input views must yield identical results to executing on explicitly `.contiguous()` tensors.
   - The operator must support `torch.compile(..., fullgraph=True)` without throwing layout exceptions or falling back out of the graph.
