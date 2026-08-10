#!/usr/bin/env bash
set -euo pipefail

# Apply the complete golden solution to /workspace.
# Keep this deterministic and safe to run once in a fresh environment.

cat << 'EOF' > /workspace/custom_op.cpp
#include <torch/extension.h>
#include <vector>

// Fixed C++ kernel with stride-aware indexing
torch::Tensor custom_weighted_sum_cpu(torch::Tensor input, torch::Tensor weights) {
    TORCH_CHECK(input.device().is_cpu(), "Input must be on CPU");
    TORCH_CHECK(weights.device().is_cpu(), "Weights must be on CPU");
    
    auto input_contig = input.contiguous();
    // In the golden solution, we make indexing stride-aware or enforce contiguity safely.
    // Here we implement stride-aware index calculation for demonstration of the fix:
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    auto output = torch::zeros({sizes[0]}, input.options());
    
    // Stride-aware computation loop
    auto input_accessor = input.accessor<float, 2>();
    auto weights_data = weights.data_ptr<float>();
    auto output_data = output.data_ptr<float>();
    
    int64_t batch_size = sizes[0];
    int64_t features = sizes[1];
    
    for (int64_t i = 0; i < batch_size; ++i) {
        float sum = 0.0f;
        for (int64_t j = 0; j < features; ++j) {
            sum += input.reshape({batch_size, features})[i][j].item<float>() * weights_data[j];
        }
        output_data[i] = sum;
    }
    return output;
}

TORCH_LIBRARY(custom_ops, m) {
    m.def("custom_weighted_sum(Tensor input, Tensor weights) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_ops, CPU, m) {
    m.impl("custom_weighted_sum", custom_weighted_sum_cpu);
}
EOF

echo "Golden solution applied successfully."

