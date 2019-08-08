#include "custom_resize_op.h"
#include "caffe2/operators/softmax_utils.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(CustomResize, CustomResizeOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    CustomResizeGradient,
    CustomResizeGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(CustomResize).NumOutputs(3);

OPERATOR_SCHEMA(CustomResizeGradient).NumOutputs(1);

class GetCustomResizeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CustomResizeGradient",
        "",
        vector<string>{I(0), I(1), O(0), O(1), O(2), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(CustomResize, GetCustomResizeGradient);
} // namespace caffe2
