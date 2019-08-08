#include "upsample_nearest_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(UpsampleNearest,
                      UpsampleNearestOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(UpsampleNearestGradient,
                      UpsampleNearestGradientOp<float, CPUContext>);

// Input: X; Output: Y
OPERATOR_SCHEMA(UpsampleNearest).NumInputs(1).NumOutputs(1);
// Input: X, dY (aka "gradOutput"); Output: dX (aka "gradInput")
OPERATOR_SCHEMA(UpsampleNearestGradient).NumInputs(2).NumOutputs(1);

class GetUpsampleNearestGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "UpsampleNearestGradient", "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(UpsampleNearest, GetUpsampleNearestGradient);

} // namespace caffe2
