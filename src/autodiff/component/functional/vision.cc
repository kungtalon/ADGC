//
// Created by kungtalon on 2022/12/25.
//
#include "autodiff/component/functional/vision.h"

namespace auto_diff {
namespace functional {

Conv2D::Conv2D(Node *input_ptr,
               Parameter *kernel_ptr,
               const std::vector<size_t> &strides,
               Graph *g,
               const std::string &name)
  : Node(NodeType::ADG_CONV2D_TYPE, {input_ptr, kernel_ptr}, name, g),
    strides_(strides) {
  set_backward_version(1);
  // second being the kernel
  // input : image features [B, H, W, Cin], kernel [Kh, Kw, Cin, Cout]
  // if input.size() == 3: use bias of size : [Cout]
  if (parents_.size() != 2) {
    throw adg_exception::OpsParentsNumException("Conv >> Conv: expect 2 parent nodes...");
  }

  if (strides_.size() != 2) {
    throw adg_exception::InvalidNodeArgumentError("Conv >> Conv: strides should have size of 2...");
  }

  for (size_t &stride : strides_) {
    if (stride < 1) {
      throw adg_exception::InvalidNodeArgumentError("Conv >> Conv: getting stride smaller than 1..");
    }
  }

  auto image_shape = parents_[0]->get_value_shape();
  kernel_shape_ = parents_[1]->get_value_shape();
  size_t h = image_shape[1];
  size_t w = image_shape[2];
  size_t in_c = image_shape[3];
  if (kernel_shape_[2] != in_c) {
    throw adg_exception::MismatchNodeValueShapeError(
      "Conv >> Conv: different channel size for image and kernel! "
        + std::to_string(kernel_shape_[2]) + " and " + std::to_string(in_c));
  }

  if (strides_[0] > h || strides_[1] > w) {
    throw adg_exception::InvalidNodeArgumentError("Conv >> Conv: strides should not bigger than input dimension..");
  }

  out_c_ = parents_[1]->get_value_shape()[parents_[1]->get_value_dim() - 1];
  out_h_ = (h - kernel_shape_[0]) / strides[0] + 1;
  out_w_ = (w - kernel_shape_[1]) / strides[1] + 1;

  value_ = DTensor({out_c_, out_h_, out_w_});
}

void Conv2D::do_forward() {
  col_kernel_ = parents_[1]->get_value().copy();
  col_kernel_.reshape({kernel_shape_[0] * kernel_shape_[1] * kernel_shape_[2], out_c_});
  // shape: [kh * kw * cin, cout]

  im2col(parents_[1]->get_value()); // shape: [B, (h - kh) * (w - kw), kh * kw * cin]

  value_ = col_image_.dot(col_kernel_); // shape: [B, (h-kh)*(w-kw), c_out]
  value_.reshape({out_h_, out_w_, out_c_});
}

DTensor Conv2D::do_backward(Node *parent_ptr) {
  return DTensor({1});
}

void Conv2D::im2col(const DTensor &input) {
  // input : [b, h, w, c]
  tensor::TensorShape shape = input.get_shape();
  std::vector<size_t> input_strides = input.get_strides();
  size_t n_channels = shape[3];
  size_t n_batchs = shape[0];

  auto col_image_values = std::vector<double>(n_batchs * out_h_ * out_w_ * n_channels);
  size_t window_size = kernel_shape_[0] * kernel_shape_[1];

  const double *input_tensor_ptr = input.get_tensor_const_ptr();
  double *col_image_ptr = &col_image_values[0];

  size_t cur_src_index, cur_dest_index = 0;
  for (size_t ib = 0; ib < shape[0]; ++ib) {
    for (size_t ih = 0; ih <= shape[1] - strides_[0]; ih += strides_[0]) {
      for (size_t iw = 0; iw <= shape[2] - strides_[1]; iw += strides_[1]) {
        cur_src_index = ib * input_strides[0] + ih * input_strides[1] + iw * input_strides[2];
        for (size_t iih = 0; iih < kernel_shape_[0]; ++iih) {
          for (size_t iiw = 0; iiw < kernel_shape_[1]; ++iiw) {
            cblas_dcopy(n_channels, input_tensor_ptr + cur_src_index, 1, col_image_ptr, window_size);
            cur_dest_index += 1;
            cur_src_index += input_strides[2]; // = n_channels
          }
          cur_src_index += input_strides[1];
        }
        cur_dest_index += (n_channels - 1) * window_size;
      }
    }
  }

  col_image_ = DTensor({n_batchs, out_h_ * out_w_, n_channels}, std::move(col_image_values));
}

}
}