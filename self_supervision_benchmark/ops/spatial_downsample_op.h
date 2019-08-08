#ifndef SPATIAL_DOWNSAMPLE_OP_H_
#define SPATIAL_DOWNSAMPLE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SpatialDownsampleOp final : public Operator<Context> {
 public:
  SpatialDownsampleOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_y_(this->template GetSingleArgument<float>("height_scale", 1.)),
        scale_x_(this->template GetSingleArgument<float>("width_scale", 1.)),
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
  float scale_y_;
  float scale_x_;
  StorageOrder order_;
  // float height_scale_;
  // float width_scale_;
  float crop_x_;
  float crop_y_;
  // Tensor tmp0_{Context::GetDeviceType()};
};

template <typename T, class Context>
class SpatialDownsampleGradientOp final : public Operator<Context> {
 public:
  SpatialDownsampleGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_y_(this->template GetSingleArgument<float>("height_scale", 1.)),
        scale_x_(this->template GetSingleArgument<float>("width_scale", 1.)),
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
  float scale_y_;
  float scale_x_;
  StorageOrder order_;
  // float height_scale_;
  // float width_scale_;
  float crop_x_;
  float crop_y_;
  Tensor tmp0_{Context::GetDeviceType()};
  Tensor tmp1_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // SPATIAL_DOWNSAMPLE_OP_H_
