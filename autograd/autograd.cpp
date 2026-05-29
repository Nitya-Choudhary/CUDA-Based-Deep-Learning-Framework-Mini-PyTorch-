#include "autograd/autograd.h"
#include <cmath>

namespace minidl {

Tensor mse_loss(const Tensor& prediction, const Tensor& target) {
    assert(prediction.shape() == target.shape());
    Tensor diff = prediction - target;
    Tensor sq = diff * diff;
    Tensor loss = sq.mean(-1);
    return loss;
}

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& target_one_hot) {
    assert(logits.shape() == target_one_hot.shape());
    Tensor probs = logits.softmax(1);
    Tensor log_probs = probs.log();
    Tensor energy = target_one_hot * log_probs;
    Tensor loss = energy.sum(1) * -1.0f;
    return loss.mean(-1);
}

} // namespace minidl
