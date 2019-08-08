#include "spatial_downsample_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    SpatialDownsample,
    SpatialDownsampleOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SpatialDownsampleGradient,
    SpatialDownsampleGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SpatialDownsample).NumOutputs(3);

OPERATOR_SCHEMA(SpatialDownsampleGradient).NumOutputs(1);

class GetSpatialDownsampleGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SpatialDownsampleGradient",
        "",
        vector<string>{I(0), O(0), O(1), O(2), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(SpatialDownsample, GetSpatialDownsampleGradient);
} // namespace caffe2
