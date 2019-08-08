#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "custom_resize_op.h"

namespace caffe2 {

namespace {

__global__ void lerpx_k(
    const float* in, int W, int H, int D, int bW, float s, float c,
    float* output) {
  CUDA_1D_KERNEL_LOOP(i, W * H * D) {
    const int x = i % W;
    const int y = (i / W) % H;
    const int z = (i / W) / H;

    const float o_x = min(max(float(0), x / s + c), float(bW-1));
    const int x0 = o_x;
    const int x1 = min(x0 + 1, bW - 1);
    const float wx = x1 - o_x;

    output[i] =  wx   *in[z*bW*H + y*bW + x0] +
                (1-wx)*in[z*bW*H + y*bW + x1];
  }
}

__global__ void lerpy_k(
    const float* in, int W, int H, int D, int bH, float s, float c,
    float* output) {
  CUDA_1D_KERNEL_LOOP(i, W * H * D) {
    const int x = i % W;
    const int y = (i / W) % H;
    const int z = (i / W) / H;

    const float o_y = min(max(float(0), y / s + c), float(bH-1));
    const int y0 = o_y;
    const int y1 = min(y0+1, bH-1);
    const float wy = y1 - o_y;

    output[i] =  wy   *in[z*W*bH + y0*W + x] +
                (1-wy)*in[z*W*bH + y1*W + x];
  }
}

extern __shared__ char blur_kernel_cache[];
template <int MR>
__global__ void tentx_k(
    const float* in, int W, int H, int D, float* out, float R) {
  assert(blockDim.z == 1);
  float* cache = reinterpret_cast<float*>(blur_kernel_cache);
  // Read an image patch into local memory
  const int bW = blockDim.x, bH = blockDim.y;
  const int x0 = bW *blockIdx.x, y0 = bH *blockIdx.y, z = blockIdx.z;
  const int dx = threadIdx.x, dy = threadIdx.y, pW = bW + 2*MR, pH = bH;
  for (int i = dx + bW *dy; i < pW*pH; i += bW * bH) {
    int x = x0 + (i % pW) - MR, y = y0 + (i / pW);
    if (x <  0) x = 0;
    if (x >= W) x = W-1;
    if (y <  0) y = 0;
    if (y >= H) y = H-1;
    cache[i] = in[x+y*W+z*W*H];
  }
  __syncthreads();
  // Aplpy the 1d filter
  int x = x0 + dx, y = y0 + dy;
  if (x < W && y < H) {
    float s = R * cache[(dx+MR) + pW * dy];
    for (int r = 1; r <= MR; r++)
      s += (R-r) * (cache[(dx+r+MR) + pW*dy] + cache[(dx-r+MR) + pW*dy]);
    float norm = ((2*MR+1)*R - MR*(MR+1));  // Closed from for the normalization
    out[x+y*W+z*W*H] = s / norm;
  }
}

template <int MR>
__global__ void tenty_k(
    const float* in, int W, int H, int D, float* out, float R) {
  assert(blockDim.z == 1);
  float * cache = reinterpret_cast<float*>(blur_kernel_cache);
  // Read an image patch into local memory
  const int bW = blockDim.x, bH = blockDim.y;
  const int x0 = bW *blockIdx.x, y0 = bH *blockIdx.y, z = blockIdx.z;
  const int dx = threadIdx.x, dy = threadIdx.y, pW = bW, pH = bH + 2*MR;
  for (int i = dx + bW *dy; i < pW*pH; i += bW * bH) {
    int x = x0 + (i % pW), y = y0 + (i / pW) - MR;
    if (x <  0) x = 0;
    if (x >= W) x = W-1;
    if (y <  0) y = 0;
    if (y >= H) y = H-1;
    cache[i] = in[x+y*W+z*W*H];
  }
  __syncthreads();
  // Aplpy the 1d filter
  int x = x0 + dx, y = y0 + dy;
  if (x < W && y < H) {
    float s = R * cache[dx + pW * (dy+MR)];
    for (int r = 1; r <= MR; r++)
      s += (R-r) * (cache[dx + pW * (dy-r+MR)] + cache[dx + pW * (dy+r+MR)]);
    float norm = ((2*MR+1)*R - MR*(MR+1));  // Closed from for the normalization
    out[x+y*W+z*W*H] = s / norm;
  }
}

template <int S>
__global__ void unlerpx_k(const float* in, int W, int H, int D, int bW,
                          float s, float c, float * output) {
  CUDA_1D_KERNEL_LOOP(i, bW * H * D) {
    const int x = i % bW;
    const int y = (i / bW) % H;
    const int z = (i / bW) / H;

    const float t_x = x * s - c;
    const int x0 = t_x - s + 0.999;  // Numerically stable ceil (if s<500)
    float sm = 0;
    for (int xx = x0; xx <= x0+S; xx++)
      if (0 <= xx && xx < W) {
        const float o_x = min(max(float(0), xx / s + c), float(bW-1));
        const float w = max(float(1)-fabs(x-o_x), float(0));
        sm += w * in[z*W*H + y*W + xx];
      }
    output[i] = sm;
  }
}

template <int S>
__global__ void unlerpy_k(const float* in, int W, int H, int D, int bH,
                          float s, float c, float* output) {
  CUDA_1D_KERNEL_LOOP(i, W * bH * D) {
    int x = i % W;
    int y = (i / W) % bH;
    int z = (i / W) / bH;

    const float t_y = y * s - c;
    const int y0 = t_y - s + 0.999;  // Numerically stable ceil (if s<500)
    float sm = 0;
    for (int yy = y0; yy < y0+S; yy++)
      if (0 <= yy && yy < H) {
        const float o_y = min(max(float(0), yy / s + c), float(bH-1));
        const float w = max(float(1)-fabs(y-o_y), float(0));
        sm += w * in[z*W*H + yy*W + x];
      }
    output[i] = sm;
    // if (i == 0) {
    //   printf("scale: %f\n", s);
    //   printf("Output[i]: %f\n", sm);
    // }
  }
}

} // namespace

static int getBS(int i, int MAX_BS = 64) {
  // Round up the the next power of two
  int r = 1;
  for (; r < i && 2*r <= MAX_BS; r *= 2) {}
  return std::min(r, MAX_BS);
}

static void tentx_gpu(
    const float* in, int W, int H, int D, float* out, float R,
    CUDAContext* context) {
  const int BX = getBS(W);
  const int BY = getBS(H, 1024 / BX);
  int MR = R;
  const int NS = (BX + 2 * MR) * BY * sizeof(float);
  if (MR == 0) {
    // caffe_copy(W * H * D, in, out);
    context->CopySameDevice<float>(W * H * D, in, out);
  }

#define CALL_K(X) else if (MR == X) tentx_k<X><<<dim3((W-1)/BX+1, (H-1)/BY+1, D), dim3(BX,BY,1), NS, context->cuda_stream()>>>(in, W, H, D, out, R)
  CALL_K(1);
  CALL_K(2);
  CALL_K(3);
  CALL_K(4);
  CALL_K(5);
  CALL_K(6);
  CALL_K(7);
  CALL_K(8);
  CALL_K(9);
  CALL_K(10);
  else {
    // LOG(WARNING) << "Filter radius too large, applying a filter of radius 10";
    tentx_k<10><<<dim3((W-1)/BX+1, (H-1)/BY+1, D), dim3(BX, BY, 1), NS, context->cuda_stream()>>>(in, W, H, D, out, R);
  }
#undef CALL_K
}


static void tenty_gpu(const float* in, int W, int H, int D, float* out, float R, CUDAContext* context) {
  const int BX = getBS(W);
  const int BY = getBS(H, 1024/BX);
  int MR = R;
  const int NS = BX * (BY + 2 * MR) * sizeof(float);
  if (MR < 1) {
    // caffe_copy(W*H*D, in, out);
    context->CopySameDevice<float>(W*H*D, in, out);
  }
  // NOLINT_NEXT_LINE
#define CALL_K(X) else if (MR == X) tenty_k<X><<<dim3((W-1)/BX+1,(H-1)/BY+1,D), dim3(BX,BY,1), NS, context->cuda_stream()>>>(in, W, H, D, out, R)
  CALL_K(1);
  CALL_K(2);
  CALL_K(3);
  CALL_K(4);
  CALL_K(5);
  CALL_K(6);
  CALL_K(7);
  CALL_K(8);
  CALL_K(9);
  CALL_K(10);
  else {
    // LOG(WARNING) << "Filter radius too large, applying a filter of radius 10";
    tenty_k<10><<<dim3((W-1)/BX+1, (H-1)/BY+1, D), dim3(BX, BY, 1), NS, context->cuda_stream()>>>(in, W, H, D, out, R);
  }
#undef CALL_K
}


static void unlerpx_gpu(const float* in, int W, int H, int D, int bW,
                        float s, float c, float * out, CUDAContext* context) {
  const int BX = getBS(bW);
  const int BY = getBS(H, 1024/BX);
  int S = ceil(2*s);
  const int NB = CAFFE_GET_BLOCKS(bW*H*D);
  const int NT = CAFFE_CUDA_NUM_THREADS;
  if (S == 0) LOG(FATAL) << "Scale factor cannot be 0";
  // NOLINT_NEXT_LINE[whitespace/operators]
#define CALL_K(X) else if (S == X) unlerpx_k<X><<<NB, NT, 0, context->cuda_stream()>>>(in, W, H, D, bW, s, c, out)
  CALL_K(1); CALL_K(2); CALL_K(3); CALL_K(4); CALL_K(5);
  CALL_K(6); CALL_K(7); CALL_K(8); CALL_K(9); CALL_K(10);
  CALL_K(11); CALL_K(12); CALL_K(13); CALL_K(14); CALL_K(15);
  CALL_K(16); CALL_K(17); CALL_K(18); CALL_K(19); CALL_K(20);
  // NOLINT_NEXT_LINE[readability/braces]
  else {
    // LOG(WARNING) << "Unlerp radius too large, using only 20 elements";
    std::cout << "Unlerp radius too large, using only 20 elements" << std::endl;
    // NOLINT_NEXT_LINE[whitespace/operators]
    unlerpx_k<20><<<NB, NT, 0, context->cuda_stream()>>>(in, W, H, D, bW, s, c, out);
  }
#undef CALL_K
}

static void unlerpy_gpu(const float* in, int W, int H, int D, int bH,
                        float s, float c, float* out, CUDAContext* context) {
  const int BX = getBS(W);
  const int BY = getBS(bH, 1024/BX);
  int S = ceil(2*s);
  const int NB = CAFFE_GET_BLOCKS(W*bH*D);
  const int NT = CAFFE_CUDA_NUM_THREADS;
  if (S == 0) LOG(FATAL) << "Scale factor cannot be 0";
  // NOLINT_NEXT_LINE[whitespace/operators]
#define CALL_K(X) else if (S == X) unlerpy_k<X><<<NB, NT, 0, context->cuda_stream()>>>(in, W, H, D, bH, s, c, out)
  CALL_K(1); CALL_K(2); CALL_K(3); CALL_K(4); CALL_K(5);
  CALL_K(6); CALL_K(7); CALL_K(8); CALL_K(9); CALL_K(10);
  CALL_K(11); CALL_K(12); CALL_K(13); CALL_K(14); CALL_K(15);
  CALL_K(16); CALL_K(17); CALL_K(18); CALL_K(19); CALL_K(20);
  // NOLINT_NEXT_LINE[readability/braces]
  else {
    LOG(WARNING) << "Unlerp radius too large, using only 20 elements";
    std::cout << "Unlerp radius too large, using only 20 elements" << std::endl;
    // NOLINT_NEXT_LINE[whitespace/operators]
    unlerpy_k<20><<<NB, NT, 0, context->cuda_stream()>>>(in, W, H, D, bH, s, c, out);
  }
#undef CALL_K
}

template <typename T>
static void Print(T &t, std::string name) {
  std::cout << name << ": ";
  for(auto x : t.sizes().vec()) {
    std::cout << x << ", " ;
  }
  std::cout << std::endl;
}

template <>
bool CustomResizeOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);     // original
  auto& Y = Input(1);     // resize like this
  // auto* top = Output(0);  // resized output
  // auto* tmp0 = Output(1);  // resized output
  // auto* tmp1 = Output(2);  // resized output

  DCHECK_EQ(X.dim(), 4);
  int N = X.dim32(0);
  int C = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);
  int D = N * C;

  int tH = Y.dim32(2);
  int tW = Y.dim32(3);

  // top->ResizeLike(Y);
  // tmp0->ResizeLike(Y);
  // tmp1->ResizeLike(Y);
  auto* top = Output(0, Y.sizes(), at::dtype<float>());
  auto* tmp0 = Output(1, Y.sizes(), at::dtype<float>());
  auto* tmp1 = Output(2, Y.sizes(), at::dtype<float>());

  const float* Xdata = X.data<float>();
  float* topData = top->mutable_data<float>();
  float* p_tmp0 = tmp0->mutable_data<float>();
  float* p_tmp1 = tmp1->mutable_data<float>();

  scale_x_ = (tW - 1.f) / (W - 1);
  scale_y_ = (tH - 1.f) / (H - 1);

  crop_x_ = 0.0;
  crop_y_ = 0.0;
  // printf("scale_x_ %f\n", scale_x_);
  // printf("scale_y_ %f\n", scale_y_);
  if (scale_x_ < 1) {
    tentx_gpu(Xdata, W, H, D, p_tmp0, float(1.)/std::max(scale_x_, float(1e-2)), &context_);
    Xdata = p_tmp0;
  }

  lerpx_k<<<CAFFE_GET_BLOCKS(tW * H * D), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      Xdata, tW, H, D, W, scale_x_, crop_y_, p_tmp1);

  if (scale_y_ < 1) {
    tenty_gpu(p_tmp1, tW, H, D, p_tmp0, float(1.)/std::max(scale_y_, float(1e-2)), &context_);
    std::swap(p_tmp1, p_tmp0);
  }

  lerpy_k<<<CAFFE_GET_BLOCKS(tW * H * D), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      p_tmp1, tW, tH, D, H, scale_y_, crop_x_, topData);
  return true;
}

template <>
bool CustomResizeGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);     // original
  auto& Y = Input(1);     // resize like this
  auto& top = Input(2);   // resized output
  auto& tmp0 = Input(3);   // resized output
  auto& tmp1 = Input(4);   // resized output
  auto& dTop = Input(5);  // resized output
  // auto* dX = Output(0);

  DCHECK_EQ(X.dim(), 4);
  int N = X.dim32(0);
  int C = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);
  int D = N * C;

  // dX->ResizeLike(X);
  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  int tH = Y.dim32(2);
  int tW = Y.dim32(3);

  scale_x_ = (tW - 1.f) / (W - 1);
  scale_y_ = (tH - 1.f) / (H - 1);
  crop_x_ = 0.0;
  crop_y_ = 0.0;

  tmp0_.ResizeLike(Y);
  tmp1_.ResizeLike(Y);

  float* p_bot = dX->mutable_data<float>();
  float* p_tmp0 = tmp0_.mutable_data<float>();
  float* p_tmp1 = tmp1_.mutable_data<float>();
  const float* p_top = dTop.data<float>();
  const float* topData = top.data<float>();
  const float* tmp0Data = tmp0.data<float>();
  const float* tmp1Data = tmp1.data<float>();

  context_.CopySameDevice<float>(Y.size(), tmp0Data, p_tmp0);
  context_.CopySameDevice<float>(Y.size(), tmp1Data, p_tmp1);

  unlerpy_gpu(p_top, tW, tH, D, H, scale_y_, crop_y_, p_tmp1, &context_);
  if (scale_y_ < 1) {
    tenty_gpu(p_tmp1, tW, H, D, p_tmp0, float(1.)/std::max(scale_y_, float(1e-2)), &context_);
    std::swap(p_tmp1, p_tmp0);
  }

  unlerpx_gpu(p_tmp1, tW, H, D, W, scale_x_, crop_x_, p_bot, &context_);
  if (scale_x_ < 1) {
    context_.CopySameDevice<float>(W*H*D, p_bot, p_tmp0);
    tentx_gpu(p_tmp0, W, H, D, p_bot, float(1.)/std::max(scale_x_, float(1e-2)), &context_);
  }
  return true;
}


REGISTER_CUDA_OPERATOR(CustomResize,
                       CustomResizeOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(CustomResizeGradient,
                       CustomResizeGradientOp<float, CUDAContext>);
} // namespace caffe2
