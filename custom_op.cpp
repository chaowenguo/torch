#include <torch/extension.h>
#include <vector>

// 存在 Bug 的初始实现：直接按连续内存读取，未考虑 strides
torch::Tensor custom_weighted_sum_cpu(torch::Tensor input, torch::Tensor weights) {
    TORCH_CHECK(input.device().is_cpu(), "Input must be on CPU");
    TORCH_CHECK(weights.device().is_cpu(), "Weights must be on CPU");
    
    auto sizes = input.sizes();
    auto output = torch::zeros({sizes[0]}, input.options());
    
    // 错误地直接使用 data_ptr，当输入是非连续 tensor（如 slice/transpose）时会导致数据错位或越界
    auto input_data = input.data_ptr<float>();
    auto weights_data = weights.data_ptr<float>();
    auto output_data = output.data_ptr<float>();
    
    int64_t batch_size = sizes[0];
    int64_t features = sizes[1];
    
    for (int64_t i = 0; i < batch_size; ++i) {
        float sum = 0.0f;
        for (int64_t j = 0; j < features; ++j) {
            // 简单直接的线性索引，没有考虑 input.strides()
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
