#ifndef UPSAMPLE_NEAREST_OP_H_
#define UPSAMPLE_NEAREST_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class UpsampleNearestOp final : public Operator<Context> {
 public:
  UpsampleNearestOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(this->template GetSingleArgument<int>("scale", 2)) {
    DCHECK_GE(scale_, 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
    return true;
  }

 protected:
  int scale_;
};

template <typename T, class Context>
class UpsampleNearestGradientOp final : public Operator<Context> {
 public:
  UpsampleNearestGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(this->template GetSingleArgument<int>("scale", 2)) {
    // Check Op parameters ...
    DCHECK_GE(scale_, 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
    return true;
  }

 protected:
  int scale_;
};

} // namespace caffe2

#endif // UPSAMPLE_NEAREST_OP_H_
