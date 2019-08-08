#include "spatial_softmax_op.h"
#include "caffe2/operators/softmax_utils.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SpatialSoftmax, SpatialSoftmaxOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SpatialSoftmaxGradient,
    SpatialSoftmaxGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SpatialSoftmax).NumOutputs(1);

OPERATOR_SCHEMA(SpatialSoftmaxGradient).NumOutputs(1);

class GetSpatialSoftmaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SpatialSoftmaxGradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(SpatialSoftmax, GetSpatialSoftmaxGradient);
} // namespace caffe2
