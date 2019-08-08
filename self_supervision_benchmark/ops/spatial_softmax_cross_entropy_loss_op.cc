#include "spatial_softmax_cross_entropy_loss_op.h"
#include "caffe2/operators/softmax_utils.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    SpatialSoftmaxCrossEntropyLoss,
    SpatialSoftmaxCrossEntropyLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SpatialSoftmaxCrossEntropyLossGradient,
    SpatialSoftmaxCrossEntropyLossGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SpatialSoftmaxCrossEntropyLoss).NumOutputs({2, 3});

OPERATOR_SCHEMA(SpatialSoftmaxCrossEntropyLossGradient).NumOutputs(1);

class GetSpatialSoftmaxCrossEntropyLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SpatialSoftmaxCrossEntropyLossGradient",
        "",
        vector<string>{I(0), I(1), I(2), O(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(
    SpatialSoftmaxCrossEntropyLoss,
    GetSpatialSoftmaxCrossEntropyLossGradient);

} // namespace caffe2

// auto& X = Input(0);               // Output                    N x 313 x X x Y
// auto& targets = Input(1);         // Ground Truth              N x 313 x X x Y
// auto& priors = Input(2);          // Priors boost              N x 1 x X x Y
// auto* avg_loss = Output(0);       // average loss as output
// auto* P = Output(1);              // softmax prob, going to be used in grad
//
// auto& X = Input(0);               // Output                    N x 313 x X x Y
// auto& targets = Input(1);         // Ground Truth              N x 313 x X x Y
// auto& priors = Input(2);          // Priors boost              N x 1 x X x Y
// auto& P = Input(3);               // softmax probabilities
// auto& d_avg_loss = Input(4);      // average loss as output
// auto* dX = Output(1);             // gradients of input
