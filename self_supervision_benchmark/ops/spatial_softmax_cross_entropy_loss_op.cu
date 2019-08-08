#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "spatial_softmax_cross_entropy_loss_op.h"

namespace caffe2 {

namespace {

__global__ void SpatialSoftmaxKernel(const int N, const int H, const int W,
    const float* Xdata, float* Pdata, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, N * H * W) {
    int C = num_classes;
    int x = index % W;
    int y = (index / W) % H;
    int i = index / W / H;

    // Subtract max on each cell for numerical reasons
    float max_val = -FLT_MAX;
    for(int c = 0; c < C; ++c) {
      int idx = i * (H * W * C) +  c * (H * W) + y * W + x;
      max_val = max(max_val, Xdata[idx]);
    }
    // Exponentiate
    float expsum = 0.0f;
    for(int c = 0; c < C; ++c) {
      int idx = i * (H * W * C) + c * (H * W) + y * W + x;
      float expx = exp(Xdata[idx] - max_val);
      Pdata[idx] = expx;
      expsum += expx;
    }
    // Normalize
    for(int c = 0; c < C; ++c) {
      int idx = i * (H * W * C) + c * (H * W) + y * W + x;
      Pdata[idx] /= expsum;
    }
  }
}


__global__ void SpatialSoftmaxCrossEntropyLossKernel(
    const int N, const int C, const int H, const int W, const float* Pdata,
    const float* targets, const float* priors, float* losses) {
  CUDA_1D_KERNEL_LOOP(i, N * H * W) {
    int x = i % W;
    int y = (i / W) % H;
    int n = i / (W * H);

    losses[i] = 0.0;
    float entropy = 0.0f;
    for (int c = 0; c < C; ++c) {
      int idx = n * (H * W * C) + c * (H * W) + y * W + x;
      entropy += targets[idx] * log(max(Pdata[idx], 1e-20f));
    }
    losses[i] = -(priors[i] * entropy);
  }
}


__global__ void SpatialSoftmaxCrossEntropyLossGradientKernel(
    const int N, const int C, const int H, const int W, const float* Pdata,
    const float* targets, const float* priors,
    const float* d_loss_data,  float* dX) {
  CUDA_1D_KERNEL_LOOP(i, N * C * H * W) {
    // to compute:
    // d_loss * prior(n, x, y) *
    // (Pdata[n, x, y, c] * sum_targets_data[n, x, y] - target[n, x, y, c])
    int x = i % W;
    int y = (i / W) % H;
    int c = (i / (W * H)) % C;
    int n = i / (W * H * C);
    float d_loss = *d_loss_data;

    int idx = n * (H * W * C) + c * (H * W) + y * W + x;
    int spatial_ind = n * (H * W) + y * W + x;
    dX[i] = 0.0;
    // dX[i] = d_loss * priors[spatial_ind] * (
    //   (Pdata[idx] * sum_targets_data[spatial_ind]) - targets[idx]);
    dX[i] = d_loss * priors[spatial_ind] * (Pdata[idx] - targets[idx]);
  }
}

} // namespace

template <typename T>
static void Print(T &t, std::string name) {
  std::cout << name << ": ";
  for(auto x : t.sizes().vec()) {
    std::cout << x << ", " ;
  }
  std::cout << std::endl;
}

template <>
bool SpatialSoftmaxCrossEntropyLossOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);               // Output                    N x 313 x X x Y
  auto& targets = Input(1);         // Ground Truth              N x 313 x X x Y
  auto& priors = Input(2);          // Priors boost              N x 1 x X x Y
  // auto* avg_loss = Output(0);       // average loss as output
  // auto* P = Output(1);              // softmax prob, going to be used in grad
  // auto* L = (OutputSize() > 2 ? Output(2) : NULL);   // elt_wise loss

  DCHECK_EQ(X.dim(), 4);
  int N = X.dim32(0);
  int C = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);

  // per spatial location loss
  // losses_.Resize(N * H * W);
  ReinitializeTensor(&losses_, {N * H * W}, at::dtype<float>().device(CUDA));
  // losses_.Resize(N * H * W);
  total_priors_ptr_.Resize(1);

  // Print(X, "X");
  // Print(targets, "targets");
  // Print(priors, "priors");

  // output probabilities
  // P->Resize(N * C * H * W);
  // P->ResizeLike(X);
  // avg_loss->Resize(vector<int64_t>());
  auto* P = Output(1, X.sizes(), at::dtype<float>());
  auto* avg_loss = Output(0, vector<int64_t>(), at::dtype<float>());
  math::Set<float, CUDAContext>(
      1, 0.0f, total_priors_ptr_.mutable_data<float>(), &context_);
  math::Set<float, CUDAContext>(
      avg_loss->size(), 0.f, avg_loss->mutable_data<float>(), &context_);
  math::Set<float, CUDAContext>(
      P->size(), 0.f, P->mutable_data<float>(), &context_);
  math::Set<float, CUDAContext>(
      losses_.size(), 0.f, losses_.mutable_data<float>(), &context_);

  const float* Xdata = X.data<float>();
  const float* targetsData = targets.data<float>();
  const float* priorsData = priors.data<float>();

  // Spatial Softmax Kernel

  // printf("classes: %d\n", num_classes_);
  SpatialSoftmaxKernel<<<CAFFE_GET_BLOCKS(N * H * W), CAFFE_CUDA_NUM_THREADS,
         0, context_.cuda_stream()>>>(
    N, H, W, Xdata, P->mutable_data<float>(), num_classes_);

  // Compute loss for each x,y location
  SpatialSoftmaxCrossEntropyLossKernel<<<CAFFE_GET_BLOCKS(N * H * W),
      CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
    N, C, H, W, P->data<float>(), targetsData, priorsData,
    losses_.mutable_data<float>());

  // sum the losses and scale
  float* avg_loss_data = avg_loss->mutable_data<float>();
  math::Sum<float, CUDAContext>(
      losses_.size(), losses_.data<float>(), avg_loss_data, &context_);
  math::Sum<float, CUDAContext>(
      priors.size(), priors.data<float>(),
      total_priors_ptr_.mutable_data<float>(), &context_);
  float total_weight = N * H * W;
  // printf("total_weight before: %f\n", total_weight);
  CUDA_CHECK(cudaMemcpyAsync(
      &total_weight,
      total_priors_ptr_.data<float>(),
      sizeof(float),
      cudaMemcpyDeviceToHost,
      context_.cuda_stream()));
  // printf("total_weight after: %f\n", total_weight);
  math::Scale<float, float, CUDAContext>(
      1, scale_ / total_weight, avg_loss_data, avg_loss_data, &context_);

  // if (L) {
  //   L->ResizeLike(losses_);
  //   L->ShareData(losses_);
  //   L->ResizeLike(priors);
  // }
  if (OutputSize() > 2) {
    OutputTensorAlias(2, losses_);
  }
  return true;
}


template<>
bool SpatialSoftmaxCrossEntropyLossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);               // logits                    N x 313 x X x Y
  auto& targets = Input(1);         // Ground Truth              N x 313 x X x Y
  auto& priors = Input(2);          // Priors boost              N x 1 x X x Y
  auto& P = Input(3);               // softmax probabilities
  auto& d_avg_loss = Input(4);      // average loss as output
  // auto* dX = Output(0);             // gradients of input

  int N = X.dim32(0);
  int C = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);

  // Print(X, "X");
  // Print(targets, "targets");
  // Print(priors, "priors");
  // Print(P, "P");
  // Print(d_avg_loss, 'd_avg_loss');

  // store the sum of targets for each (n, x, y)
  // buff_.Resize(N * H * W);
  ReinitializeTensor(&buff_, {N * H * W}, at::dtype<float>().device(CUDA));
  // dX->Resize(N * C * H * W);
  // dX->ResizeLike(X);
  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  total_priors_ptr_.Resize(1);

  const float* targetsData = targets.data<float>();
  const float* Pdata = P.data<float>();
  const float* priorsData = priors.data<float>();

  math::Set<float, CUDAContext>(
      1, 0.0f, total_priors_ptr_.mutable_data<float>(), &context_);
  math::Sum<float, CUDAContext>(
      priors.size(), priors.data<float>(),
      total_priors_ptr_.mutable_data<float>(), &context_);
  float total_weight = N * H * W;
  CUDA_CHECK(cudaMemcpyAsync(
      &total_weight,
      total_priors_ptr_.data<float>(),
      sizeof(float),
      cudaMemcpyDeviceToHost,
      context_.cuda_stream()));

  // // Compute the sum of targets for each (n, x, y) location
  // SumTargetsKernel<<<CAFFE_GET_BLOCKS(N * H * W), CAFFE_CUDA_NUM_THREADS,
  //     0, context_.cuda_stream()>>>(
  //   N, C, H, W, targetsData, buff_.mutable_data<float>());

  // Compute the gradient now
  // const float* Bdata = buff_.data<float>();
  SpatialSoftmaxCrossEntropyLossGradientKernel
      <<<CAFFE_GET_BLOCKS(N * C * H * W), CAFFE_CUDA_NUM_THREADS,
         0, context_.cuda_stream()>>>(
    N, C, H, W, Pdata, targetsData, priorsData, d_avg_loss.data<float>(),
    dX->mutable_data<float>());
  math::Scale<float, float, CUDAContext>(
    dX->size(), scale_ / total_weight, dX->data<float>(), dX->mutable_data<float>(),
    &context_);
  return true;
}


REGISTER_CUDA_OPERATOR(SpatialSoftmaxCrossEntropyLoss,
                       SpatialSoftmaxCrossEntropyLossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SpatialSoftmaxCrossEntropyLossGradient,
                       SpatialSoftmaxCrossEntropyLossGradientOp<float, CUDAContext>);
} // namespace caffe2
