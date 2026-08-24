#!/usr/bin/env bash
set -euo pipefail

cat << 'EOF' > /workspace/custom_op.cpp
#include <torch/extension.h>
#include <vector>

// Fixed C++ kernel with stride-aware indexing and meta registration
torch::Tensor sparse_gather_scale_cpu(torch::Tensor features, torch::Tensor indices, torch::Tensor scales) {
    TORCH_CHECK(features.device().is_cpu(), "Features must be on CPU");
    TORCH_CHECK(indices.device().is_cpu(), "Indices must be on CPU");
    TORCH_CHECK(scales.device().is_cpu(), "Scales must be on CPU");
    
    auto num_indices = indices.size(0);
    auto hidden_dim = features.size(1);
    auto strides = features.strides();
    
    auto output = torch::zeros({num_indices, hidden_dim}, features.options());
    
    auto feat_data = features.data_ptr<float>();
    auto idx_data = indices.data_ptr<int64_t>();
    auto scale_data = scales.data_ptr<float>();
    auto out_data = output.data_ptr<float>();
    
    int64_t stride_0 = strides[0];
    int64_t stride_1 = strides[1];
    
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        float sc = scale_data[i];
        for (int64_t j = 0; j < hidden_dim; ++j) {
            out_data[i * hidden_dim + j] = feat_data[idx * stride_0 + j * stride_1] * sc;
        }
    }
    return output;
}

torch::Tensor sparse_gather_scale_meta(torch::Tensor features, torch::Tensor indices, torch::Tensor scales) {
    TORCH_CHECK(features.dim() == 2, "Features must be 2D");
    TORCH_CHECK(indices.dim() == 1, "Indices must be 1D");
    TORCH_CHECK(scales.dim() == 1, "Scales must be 1D");
    auto num_indices = indices.size(0);
    auto hidden_dim = features.size(1);
    return torch::empty({num_indices, hidden_dim}, features.options());
}

TORCH_LIBRARY(custom_ops, m) {
    m.def("sparse_gather_scale(Tensor features, Tensor indices, Tensor scales) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_ops, CPU, m) {
    m.impl("sparse_gather_scale", sparse_gather_scale_cpu);
}

TORCH_LIBRARY_IMPL(custom_ops, Meta, m) {
    m.impl("sparse_gather_scale", sparse_gather_scale_meta);
}
EOF

echo "Golden solution applied successfully."
