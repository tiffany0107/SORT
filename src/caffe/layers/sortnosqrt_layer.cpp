#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/sortnosqrt_layer.hpp"
#include "caffe/util/math_functions.hpp"



namespace caffe {

template <typename Dtype>
void SortnosqrtLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[1]->shape() == bottom[0]->shape());
  
//   sqrt_shift=0.01;
}

template <typename Dtype>
void SortnosqrtLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*bottom[0]);
// Blobs 'bottom_relu' and 'bottom_relu_1' store the value of ReLU(bottom[0]) and ReLU(bottom[1])
  bottom_relu.Reshape(bottom[0]->shape());
  bottom_relu_1.Reshape(bottom[0]->shape());

// Blob 'bottom_gradient' and 'bottom_gradient_1' store the gradient of production part for bottom[0] and bottom[1].
//  bottom_gradient.Reshape(bottom[0]->shape());
//  bottom_gradient_1.Reshape(bottom[0]->shape());

// Blob 'after_prod' stores the value of ReLU(bottom[0])*ReLU(bottom[1]).
  after_prod.Reshape(bottom[0]->shape());

// Blob 'after_sqrt' stores the value of (ReLU(bottom[0])*ReLU(bottom[1])+shift)^0.5.
//  after_sqrt.Reshape(bottom[0]->shape());

// Blob 'gradient for prod' stores part of the gradient equation, which is 0.5(ReLU(bottom[0])*ReLU(bottom[1])+shift)^(-0.5).
//  gradient_for_prod.Reshape(bottom[0]->shape());
  
}

template <typename Dtype>
void SortnosqrtLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_data_1 = bottom[1]->cpu_data();
  Dtype* bottom_relu_data = bottom_relu.mutable_cpu_data();
  Dtype* bottom_relu_data_1 = bottom_relu_1.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    bottom_relu_data[i]=std::max(bottom_data[i],Dtype(0));
    bottom_relu_data_1[i]=std::max(bottom_data_1[i],Dtype(0));
  }
  caffe_mul(count, bottom_relu.cpu_data(), bottom_relu_1.cpu_data(), after_prod.mutable_cpu_data());
  caffe_add(count, bottom_data, bottom_data_1, top_data);
  caffe_add(count, top[0]->cpu_data(), after_prod.cpu_data(), top_data);
} 

template <typename Dtype>
void SortnosqrtLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_data_1 = bottom[1]->cpu_data();

    Dtype* bottom_relu_data = bottom_relu.mutable_cpu_data();
    Dtype* bottom_relu_data_1 = bottom_relu_1.mutable_cpu_data();


    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype* bottom_diff_1 = bottom[1]->mutable_cpu_diff();

    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_relu_data[i]=std::max(bottom_data[i],Dtype(0));
      bottom_relu_data_1[i]=std::max(bottom_data_1[i],Dtype(0));
    }
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (1.0+bottom_relu_data_1[i]*(bottom_data[i] > 0));
      bottom_diff_1[i] = top_diff[i] * (1.0+bottom_relu_data[i]*(bottom_data_1[i] > 0));    
    }
 
  }
}


#ifdef CPU_ONLY
STUB_GPU(SortnosqrtLayer);
#endif

INSTANTIATE_CLASS(SortnosqrtLayer);
REGISTER_LAYER_CLASS(Sortnosqrt);

}  // namespace caffe
