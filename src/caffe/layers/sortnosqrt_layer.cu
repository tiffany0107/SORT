#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/sortnosqrt_layer.hpp"
#include "caffe/util/math_functions.hpp"



namespace caffe {

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out
    ) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : 0;
  }
}

template <typename Dtype>
void SortnosqrtLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_data_1 = bottom[1]->gpu_data();
  Dtype* bottom_relu_data = bottom_relu.mutable_gpu_data();
  Dtype* bottom_relu_data_1 = bottom_relu_1.mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom_relu_data);
  
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data_1, bottom_relu_data_1);
  
  caffe_gpu_mul(count, bottom_relu.gpu_data(), bottom_relu_1.gpu_data(), after_prod.mutable_gpu_data());
  caffe_gpu_add(count, bottom_data, bottom_data_1, top_data);
  caffe_gpu_add(count, top[0]->gpu_data(), after_prod.gpu_data(), top_data);
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, const Dtype* in_data_1, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (1.0 + in_data_1[index]*(in_data[index] > 0));
  }
}

template <typename Dtype>
void SortnosqrtLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_data_1 = bottom[1]->gpu_data();

    Dtype* bottom_relu_data = bottom_relu.mutable_gpu_data();
    Dtype* bottom_relu_data_1 = bottom_relu_1.mutable_gpu_data();

    const Dtype* top_diff = top[0]->gpu_diff();

    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* bottom_diff_1 = bottom[1]->mutable_gpu_diff();

    const int count = bottom[0]->count();
    ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom_relu_data);
    ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data_1, bottom_relu_data_1);

    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_relu_data_1, bottom_diff);    
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data_1, bottom_relu_data, bottom_diff_1);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SortnosqrtLayer);

}  // namespace caffe
