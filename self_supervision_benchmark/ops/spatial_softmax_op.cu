#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "spatial_softmax_op.h"

namespace caffe2 {

namespace {

__global__ void SpatialSoftmaxKernel(const int N,
    const int H, const int W, const float* Xdata, float* Pdata,
    const int num_classes) {
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

__global__ void SumProbsKernel(const int N,
    const int W, const int H, const float* Ydata, const float* dYdata,
    float* sum_probs_data, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N * W * H) {
    // printf("Inside Second kernel: i: %d\n", i);
    int C = num_classes;
    int x = i % W;
    int y = (i / W) % H;
    int n = i / (W * H);

    sum_probs_data[i] = 0.0;
    for(int c = 0; c < num_classes; ++c) {
      int idx = n * (H * W * C) + c * (H * W) + y * W + x;
      sum_probs_data[i] += (Ydata[idx] * dYdata[idx]);
    }
  }
}

__global__ void SubSumKernel(
    const int N, const int W, const int H,
    const float* sum_probs_data, float* dXdata, const int C) {
  CUDA_1D_KERNEL_LOOP(i, N * C * W * H) {
    int x = i % W;
    int y = (i / W) % H;
    int n = i / W / H / C;
    int idx = n * (H * W) + y * W + x;
    dXdata[i] = (dXdata[i] - sum_probs_data[idx]);
  }
}

} // namespace

// template <typename T>
// static void Print(T &t, std::string name) {
//   std::cout << name << ": ";
//   for(auto x : t.sizes().vec()) {
//     std::cout << x << ", " ;
//   }
//   std::cout << std::endl;
// }

template <>
bool SpatialSoftmaxOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);         // Logits
  // auto* P = Output(0);        // softmax probability, going to be re-used in gradient

  DCHECK_EQ(X.dim(), 4);
  int N = X.dim32(0);
  int C = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);

  // P->Resize(N * C * H * W);
  auto* P = Output(0, {N * C * H * W}, at::dtype<float>());

  math::Set<float, CUDAContext>(
      P->size(), 0.f, P->mutable_data<float>(), &context_);

  const float* Xdata = X.data<float>();
  // Spatial Softmax Kernel
  SpatialSoftmaxKernel
      <<<CAFFE_GET_BLOCKS(N * H * W), CAFFE_CUDA_NUM_THREADS,
         0, context_.cuda_stream()>>>(
    N, H, W, Xdata, P->mutable_data<float>(), num_classes_);
  return true;
}

template <>
bool SpatialSoftmaxGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);   // Probabilities from softmax
  auto& dY = Input(1);  // Gradient w.r.t. avg loss
  // auto* dX = Output(0);

  DCHECK_EQ(Y.dim(), 4);
  int N = Y.dim32(0);
  int C = Y.dim32(1);
  int H = Y.dim32(2);
  int W = Y.dim32(3);

  // dX->ResizeLike(Y);
  auto* dX = Output(0, Y.sizes(), at::dtype<float>());

  if (sum_probs_.size() != N * H * W) {
    // sum_probs_.Resize(N * H * W);
    ReinitializeTensor(&sum_probs_, {N * H * W}, at::dtype<float>().device(CUDA));
  }

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  float* sum_probs_data = sum_probs_.mutable_data<float>();
  math::Set<float, CUDAContext>(
      sum_probs_.size(), 0.0f, sum_probs_data, &context_);

  // Complete math:
  // J_ij = h_i (delta_ij - h_j)
  // d x_i = sum_j d h_ij = sum_j J_ij * dy_j
  //       = sum_j h_i (delta_ij - h_j) * dy_j
  //       = h_i dy_i - (sum_j h_i h_j dy_j)
  //       = h_i dy_i - h_i sum_j h_j dy_j
  /*
    // step 1: dx = dy * y
    math::Mul<float, CUDAContext>(Y.size(), dYdata, Ydata, dXdata, &context_);

    // step 2: s = sum_i y_i * dy_i
    SumProbsKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
                     context_.cuda_stream()>>>(
                       N, A, W, H, Ydata, dYdata, sum_probs_data, num_classes_);
    // step 3: dx -= y * s
    //         dx = y * dy - y * dy * s
    //         dx = y * dy - y * sum_i y_i * dy_i
    SubSumKernel<<<CAFFE_GET_BLOCKS(Y.size()), CAFFE_CUDA_NUM_THREADS, 0,
                    context_.cuda_stream()>>>(
      N, A, W, H, sum_probs_.data<float>(), dXdata, num_classes_);
  */
  // Step 0: dx = dy
  context_.CopySameDevice<float>(Y.size(), dYdata, dXdata);
  // Step 1: s = Sum(dY[j] * Y[j])
  SumProbsKernel<<<CAFFE_GET_BLOCKS(N * H * W), CAFFE_CUDA_NUM_THREADS, 0,
                   context_.cuda_stream()>>>(
    N, W, H, Ydata, dYdata, sum_probs_data, num_classes_);
  // Step 2: dX[i] = dX[i] - s
  SubSumKernel<<<CAFFE_GET_BLOCKS(N * C * H * W), CAFFE_CUDA_NUM_THREADS, 0,
                  context_.cuda_stream()>>>(
    N, W, H, sum_probs_.data<float>(), dXdata, num_classes_);

  // Step 3: dX[i] = Y[i] * dX[i]
  math::Mul<float, CUDAContext>(Y.size(), dXdata, Ydata, dXdata, &context_);

  return true;
}


REGISTER_CUDA_OPERATOR(SpatialSoftmax,
                       SpatialSoftmaxOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SpatialSoftmaxGradient,
                       SpatialSoftmaxGradientOp<float, CUDAContext>);
} // namespace caffe2
