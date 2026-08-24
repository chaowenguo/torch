#include <torch/extension.h>
#include <vector>

// Buggy initial implementation: assumes contiguous embedding storage,
// ignores tensor strides, and lacks meta/fake registration.
torch::Tensor sparse_gather_scale_cpu(torch::Tensor features, torch::Tensor indices, torch::Tensor scales) {
    TORCH_CHECK(features.device().is_cpu(), "Features must be on CPU");
    TORCH_CHECK(indices.device().is_cpu(), "Indices must be on CPU");
    TORCH_CHECK(scales.device().is_cpu(), "Scales must be on CPU");
    
    auto num_indices = indices.size(0);
    auto hidden_dim = features.size(1);
    
    auto output = torch::zeros({num_indices, hidden_dim}, features.options());
    
    auto feat_data = features.data_ptr<float>();
    auto idx_data = indices.data_ptr<int64_t>();
    auto scale_data = scales.data_ptr<float>();
    auto out_data = output.data_ptr<float>();
    
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        float sc = scale_data[i];
        for (int64_t j = 0; j < hidden_dim; ++j) {
            // Bug: assumes contiguous memory layout, ignoring features.strides()
            out_data[i * hidden_dim + j] = feat_data[idx * hidden_dim + j] * sc;
        }
    }
    return output;
}

TORCH_LIBRARY(custom_ops, m) {
    m.def("sparse_gather_scale(Tensor features, Tensor indices, Tensor scales) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_ops, CPU, m) {
    m.impl("sparse_gather_scale", sparse_gather_scale_cpu);
}
