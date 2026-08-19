set -euo pipefail

cat << 'EOF' > /workspace/custom_op.cpp
#include <torch/extension.h>
#include <vector>

torch::Tensor custom_weighted_sum_cpu(torch::Tensor input, torch::Tensor weights) {
    TORCH_CHECK(input.device().is_cpu(), "Input must be on CPU");
    TORCH_CHECK(weights.device().is_cpu(), "Weights must be on CPU");
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");
    TORCH_CHECK(weights.dim() == 1, "Weights must be a 1D tensor");
    TORCH_CHECK(sizes[1] == weights.size(0), "Feature dimension must match weights size");

    auto output = torch::zeros({sizes[0]}, input.options());
    
    auto input_data = input.data_ptr<float>();
    auto weights_data = weights.data_ptr<float>();
    auto output_data = output.data_ptr<float>();
    
    int64_t batch_size = sizes[0];
    int64_t features = sizes[1];
    int64_t stride_0 = strides[0];
    int64_t stride_1 = strides[1];
    
    for (int64_t i = 0; i < batch_size; ++i) {
        float sum = 0.0f;
        for (int64_t j = 0; j < features; ++j) {
            sum += input_data[i * stride_0 + j * stride_1] * weights_data[j];
        }
        output_data[i] = sum;
    }
    return output;
}

torch::Tensor custom_weighted_sum_meta(torch::Tensor input, torch::Tensor weights) {
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");
    TORCH_CHECK(weights.dim() == 1, "Weights must be a 1D tensor");
    auto sizes = input.sizes();
    return torch::empty({sizes[0]}, input.options());
}

TORCH_LIBRARY(custom_ops, m) {
    m.def("custom_weighted_sum(Tensor input, Tensor weights) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_ops, CPU, m) {
    m.impl("custom_weighted_sum", custom_weighted_sum_cpu);
}

TORCH_LIBRARY_IMPL(custom_ops, Meta, m) {
    m.impl("custom_weighted_sum", custom_weighted_sum_meta);
}
EOF

echo "Golden solution applied successfully."
