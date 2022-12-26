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
  // input : image features [B, Cin, H, W], kernel [Cout, Cin, Kh, Kw]
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
  if (kernel_shape_[1] != in_c) {
    throw adg_exception::MismatchNodeValueShapeError(
      "Conv >> Conv: different channel size for image and kernel! "
        + std::to_string(kernel_shape_[1]) + " and " + std::to_string(in_c));
  }

  if (strides_[0] > h || strides_[1] > w) {
    throw adg_exception::InvalidNodeArgumentError("Conv >> Conv: strides should not bigger than input dimension..");
  }

  out_c_ = kernel_shape_[0];
  out_h_ = (h - kernel_shape_[2]) / strides[0] + 1;
  out_w_ = (w - kernel_shape_[3]) / strides[1] + 1;

  value_ = DTensor({parents_[0]->get_value_shape()[0], out_c_, out_h_, out_w_});
}

void Conv2D::do_forward() {
  col_kernel_ = parents_[1]->get_value().copy();
  col_kernel_.reshape({kernel_shape_[0], kernel_shape_[1] * kernel_shape_[2] * kernel_shape_[3]});
  // shape: [cout, kh * kw * cin]

  im2col(parents_[0]->get_value(),
         col_image_,
         kernel_shape_[2],
         kernel_shape_[3],
         strides_[0],
         strides_[1]); // shape: [B * (h - kh) * (w - kw), kh * kw * cin]

  value_ = col_image_.dot(col_kernel_.t()); // shape: [B * (h-kh)*(w-kw), c_out]
  value_.reshape({parents_[0]->get_value_shape()[0], out_h_, out_w_, out_c_});
}

DTensor Conv2D::do_backward(Node *parent_ptr) {
  // grad shape [b, h, w, cout]
  DTensor grad = get_grad().copy();
  size_t n_batch = grad.get_shape()[0];

  if (parent_ptr == parents_[1]) {
    grad.reshape({n_batch * out_h_ * out_w_, out_c_}); // [b * h*w, cout]
    return grad.t().dot(col_image_);  // [cout, kh*kw*cin]
  }

  grad.reshape({n_batch, out_h_ * out_w_, out_c_});
  grad = grad.transpose(1, 2);
  grad.reshape({n_batch, out_c_, out_h_, out_w_});
  // if parent is the image
  // dilate and pad the original
  auto transformed_grad = tensor::pad2d(
    tensor::dilate2d(grad, {strides_[0] - 1, strides_[1] - 1}),
    {kernel_shape_[0] - 1, kernel_shape_[1] - 1});

  transformed_grad.reshape({n_batch, out_c_, out_h_ * out_w_});
  transformed_grad = transformed_grad.transpose(1, 2);
  transformed_grad.reshape({n_batch, out_h_, out_w_, out_c_});
  // then, reverse each column of col_kernel
  tensor::reverse(col_kernel_, 1);  // shape: [cout, kh * kw * cin]

  // do the convolution between grad and col_kernel
  im2col_hwc(transformed_grad,
             col_image_,
             kernel_shape_[2],
             kernel_shape_[3],
             1,
             1);  // shape: [B, ]
  DTensor result = col_image_.dot(col_kernel_.t()); // [B * h * w, kh * kw * cout]
  result.reshape(parent_ptr->get_value_shape());
  return result;
}

void Conv2D::im2col_hwc(const DTensor &input,
                        DTensor &output,
                        const size_t &kh,
                        const size_t &kw,
                        const size_t &sh,
                        const size_t &sw) {
  // input : [b, h, w, c]
  tensor::TensorShape shape = input.get_shape();
  std::vector<size_t> input_strides = input.get_strides();
  size_t n_channels = shape[3];
  size_t n_batchs = shape[0];

  size_t window_size = kh * kw;

  auto col_image_values =
    std::vector<double>(n_batchs * out_h_ * out_w_ * window_size * n_channels);

  const double *input_tensor_ptr = input.get_tensor_const_ptr();
  double *col_image_ptr = &col_image_values[0];

  size_t ib, ih, iw, iih, iiw;
  size_t cur_src_index, in_window_index;
  for (ib = 0; ib < shape[0]; ++ib) {
    for (ih = 0; ih <= shape[1] - sh; ih += sh) {
      for (iw = 0; iw <= shape[2] - sw; iw += sw) {
        cur_src_index = ib * input_strides[0] + ih * input_strides[1] + iw * input_strides[2];
        in_window_index = 0;
        while (in_window_index < window_size) {
          iih = in_window_index / kw;
          iiw = in_window_index % kw;
          cblas_dcopy(n_channels,
                      input_tensor_ptr + cur_src_index + iih * input_strides[1] + iiw * input_strides[2],
                      1,
                      col_image_ptr + in_window_index,
                      window_size);
          ++in_window_index;
        }
        col_image_ptr += n_channels * window_size;
      }
    }
  }

  output = DTensor({n_batchs * out_h_ * out_w_, kw * kh * n_channels},
                   std::move(col_image_values));
}

void Conv2D::im2col_chw(const DTensor &input,
                        DTensor &output,
                        const size_t &kh,
                        const size_t &kw,
                        const size_t &sh,
                        const size_t &sw) {
  // input : [b, c, h, w]
  tensor::TensorShape shape = input.get_shape();
  std::vector<size_t> input_strides = input.get_strides();
  size_t n_channels = shape[1];
  size_t n_batchs = shape[0];

  size_t window_size = kh * kw;

  auto col_image_values =
    std::vector<double>(n_batchs * out_h_ * out_w_ * window_size * n_channels);

  const double *input_tensor_ptr = input.get_tensor_const_ptr();
  double *col_image_ptr = &col_image_values[0];

  size_t ib, ic, ih, iw, iih, iiw;
  size_t cur_src_index, in_window_index, cur_dest_index;
  size_t dest_stride = window_size * n_channels;
  for (ib = 0; ib < shape[0]; ++ib) {
    for (ic = 0; ic < shape[1]; ++ic) {
      cur_dest_index = ic * window_size;
      cur_src_index = ib * input_strides[0] + ic * input_strides[1];
      for (ih = 0; ih <= shape[2] - sh; ih += sh) {
        for (iw = 0; iw <= shape[3] - sw; iw += sw) {
          in_window_index = 0;
          while (in_window_index < window_size) {
            iih = in_window_index / kw;
            iiw = in_window_index % kw;
            *(col_image_ptr + cur_dest_index + in_window_index) =
              *(input_tensor_ptr + cur_src_index + (ih + iih) * input_strides[1] + (iw + iiw) * input_strides[2]);
            ++in_window_index;
          }
          cur_dest_index += dest_stride;
        }
      }
    }
  }

  output = DTensor({n_batchs * out_h_ * out_w_, kw * kh * n_channels},
                   std::move(col_image_values));
}

}
}