//
// Created by kungtalon on 2022/12/27.
//


#include "autodiff/component/functional/normalization.h"

namespace auto_diff {
namespace functional {

BatchNorm2D::BatchNorm2D(Node *input_ptr,
                         Parameter *gamma,
                         Parameter *beta,
                         const double &epsilon,
                         const double &momentum,
                         Graph *g,
                         const std::string &name)
  : Node(NodeType::ADG_MATMUL_TYPE, {input_ptr, gamma, beta}, name, g),
    momentum_(momentum),
    epsilon_(epsilon),
    size_bhw_(0) {
  set_backward_version(1);
  // input shape: [b, c, h, w]
  if (gamma->get_value_dim() != 1 || beta->get_value_dim() != 1) {
    throw adg_exception::InvalidNodeArgumentError(
      "BatchNorm2D: expect affine transform gamma and beta to be vector...");
  }

  if (gamma->get_value_size() != input_ptr->get_value().get_shape(1) ||
    beta->get_value_size() != gamma->get_value_size()) {
    throw adg_exception::MismatchNodeValueShapeError(
      "BatchNorm2D: expect the affine transform gamma and beta have the same length of channels as the image...");
  }

  cached_tensors_.resize(2);

  moving_mean_ = DTensor(gamma->get_value_shape());
  moving_var_ = DTensor(gamma->get_value_shape());
}

void BatchNorm2D::do_forward() {
  DTensor input_tensor = parents_[0]->get_value();

  DTensor mean, var;

  if (graph_->stage() == GraphStageFlag::train) {
    if (!size_bhw_) {
      size_bhw_ = input_tensor.get_size() / input_tensor.get_shape(1); // b*h*w
    }
    assert(size_bhw_ == input_tensor.get_size() / input_tensor.get_shape(1));

    mean = input_tensor.sum(3).sum(2).sum(0).div(size_bhw_);
    DTensor sample_var_all = tensor::square(tensor::add_vec(input_tensor, -mean, 1));
    var = sample_var_all.sum(3).sum(2).sum(0).div(size_bhw_);

    // update moving averages
    moving_mean_ = tensor::add(moving_mean_.multiply(momentum_),
                               mean.multiply(1 - momentum_));
    moving_var_ = tensor::add(moving_var_.multiply(momentum_),
                              var.multiply(1 - momentum_));
  } else {
    // use Bessel correction
    mean = moving_mean_.multiply(size_bhw_ / (size_bhw_ - 1));
    var = moving_var_.multiply(size_bhw_ / (size_bhw_ - 1));
  }

  // get the normalized input, (x - mu) / sqrt(var + eps)
  DTensor inverse_std_err = tensor::sqrt(var.add(epsilon_));
  inverse_std_err.map([](double &val) { val = 1. / val; });
  DTensor input_unbiased = tensor::add_vec(input_tensor, -mean, 1);
  DTensor input_normed = tensor::pmul_vec(input_unbiased, inverse_std_err, 1);

  // cache the forwarded tensors for backward
  if (graph_->stage() == GraphStageFlag::train) {
    cached_tensors_[0] = input_normed;
    cached_tensors_[1] = inverse_std_err;
  }

  // output: gamma * x_ + beta
  value_ = tensor::add_vec(
    tensor::pmul_vec(input_normed, parents_[1]->get_value(), 1), // gamma * x
    parents_[2]->get_value(),  // beta
    1);
}

DTensor BatchNorm2D::do_backward(Node *parent_ptr) {
  DTensor grad = get_grad();

  if (parent_ptr == parents_[1]) {
    // dgamma = grad pmul
    return grad.multiply(cached_tensors_[0]).sum(3).sum(2).sum(0);
  }

  if (parent_ptr == parents_[2]) {
    // dbeta = sum(grad)
    return grad.sum(3).sum(2).sum(0);
  }

  // dx
  DTensor cached_xn = cached_tensors_[0];
  DTensor cached_inverse_se = cached_tensors_[1];

  // https://zhuanlan.zhihu.com/p/45614576
  DTensor d_xn = tensor::pmul_vec(grad, parents_[1]->get_value(), 1);
  DTensor d_x = d_xn.multiply(size_bhw_);
  d_x = tensor::add_vec(d_x, -d_xn.sum(3).sum(2).sum(0), 1);
  DTensor d_xxn = cached_xn.multiply(d_xn).sum(3).sum(2).sum(0);
  d_x = d_x.sub(tensor::pmul_vec(cached_xn, d_xxn, 1));
  d_x = tensor::pmul_vec(d_x.div(size_bhw_), cached_inverse_se, 1);
  return d_x;
}

DTensor BatchNorm2D::get_moving_mean() {
  if (this != unique_ptr_) {
    BatchNorm2D *real_ptr = dynamic_cast<BatchNorm2D *> (unique_ptr_);
    return real_ptr->moving_mean_;
  }
  return moving_mean_;
}

DTensor BatchNorm2D::get_moving_var() {
  if (this != unique_ptr_) {
    BatchNorm2D *real_ptr = dynamic_cast<BatchNorm2D *> (unique_ptr_);
    return real_ptr->moving_var_;
  }
  return moving_var_;
}

}
}