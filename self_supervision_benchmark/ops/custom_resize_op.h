#ifndef CUSTOM_RESIZE_OP_H_
#define CUSTOM_RESIZE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class CustomResizeOp final : public Operator<Context> {
 public:
  CustomResizeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  StorageOrder order_;
  float scale_x_;
  float scale_y_;
  float crop_x_;
  float crop_y_;
  // Tensor tmp0_{Context::GetDeviceType()};
};

template <typename T, class Context>
class CustomResizeGradientOp final : public Operator<Context> {
 public:
  CustomResizeGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  StorageOrder order_;
  float scale_x_;
  float scale_y_;
  float crop_x_;
  float crop_y_;
  Tensor tmp0_{Context::GetDeviceType()};
  Tensor tmp1_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // CUSTOM_RESIZE_OP_H_
