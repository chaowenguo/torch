#include <torch/extension.h>
#include <vector>

// Buggy initial implementation: assumes contiguous memory and ignores strides,
// and lacks meta/fake tensor registration entirely.
torch::Tensor custom_weighted_sum_cpu(torch::Tensor input, torch::Tensor weights) {
    TORCH_CHECK(input.device().is_cpu(), "Input must be on CPU");
    TORCH_CHECK(weights.device().is_cpu(), "Weights must be on CPU");
    
    auto sizes = input.sizes();
    auto output = torch::zeros({sizes[0]}, input.options());
    
    // Incorrectly assumes contiguous layout, causing data corruption on non-contiguous tensors
    auto input_data = input.data_ptr<float>();
    auto weights_data = weights.data_ptr<float>();
    auto output_data = output.data_ptr<float>();
    
    int64_t batch_size = sizes[0];
    int64_t features = sizes[1];
    
    for (int64_t i = 0; i < batch_size; ++i) {
        float sum = 0.0f;
        for (int64_t j = 0; j < features; ++j) {
            // Hardcoded linear offset assuming standard contiguous strides
            sum += input_data[i * features + j] * weights_data[j];
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
