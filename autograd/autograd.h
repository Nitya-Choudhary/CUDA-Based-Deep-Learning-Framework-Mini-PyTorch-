#pragma once

#include "tensor/tensor.h"

namespace minidl {

Tensor mse_loss(const Tensor& prediction, const Tensor& target);
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& target_one_hot);

} // namespace minidl
