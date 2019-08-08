#ifndef SPATIAL_SOFTMAX_CROSS_ENTROPY_LOSS_OP_H_
#define SPATIAL_SOFTMAX_CROSS_ENTROPY_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SpatialSoftmaxCrossEntropyLossOp final : public Operator<Context> {
 public:
  SpatialSoftmaxCrossEntropyLossOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(this->template GetSingleArgument<float>("scale", 1.)),
        num_classes_(this->template GetSingleArgument<int>("num_classes", 313)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float scale_;
  int num_classes_;
  StorageOrder order_;
  // Tensor losses_{Context::GetDeviceType()}; // Per example loss
  Tensor losses_; // Per example loss
  Tensor total_priors_ptr_{Context::GetDeviceType()};
};

template <typename T, class Context>
class SpatialSoftmaxCrossEntropyLossGradientOp final : public Operator<Context> {
 public:
  SpatialSoftmaxCrossEntropyLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(this->template GetSingleArgument<float>("scale", 1.)),
        num_classes_(this->template GetSingleArgument<int>("num_classes", 313)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float scale_;
  int num_classes_;
  StorageOrder order_;
  Tensor total_priors_ptr_{Context::GetDeviceType()};
  // Tensor buff_{Context::GetDeviceType()};
  Tensor buff_;
};

} // namespace caffe2

#endif // SPATIAL_SOFTMAX_CROSS_ENTROPY_LOSS_OP_H_
