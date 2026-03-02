#include "linalg.h"
#include "tensor.h"
#include <optim.h>

int adam_step(float lr, float beta1, float beta2, float eps, size_t t,
              const Tensor *grad, Tensor *param, Tensor *m, Tensor *v) {

  if (grad == NULL || param == NULL || m == NULL || v == NULL) {
    return 1;
  }
  if (!shape_is_equal(param->shape, grad->shape) ||
      !shape_is_equal(param->shape, m->shape) ||
      !shape_is_equal(param->shape, v->shape)) {
    return 2;
  }

  if (grad->dtype != DTYPE_FLOAT32 || param->dtype != DTYPE_FLOAT32 ||
      m->dtype != DTYPE_FLOAT32 || v->dtype != DTYPE_FLOAT32) {
    return 3;
  }
  tensor_scale_float(m, beta1);
  tensor_scale_and_add(m, (1.0f - beta1), grad);
  tensor_scale_float(v, beta2);
  tensor_square_scale_and_add(v, (1.0f - beta2), grad);
  float *param_data = (float *)param->data;
  float *m_data = (float *)m->data;
  float *v_data = (float *)v->data;
  float beta1_t = powf(beta1, t);
  float beta2_t = powf(beta2, t);
  for (size_t i = 0; i < param->size; ++i) {
    param_data[i] -=
        lr * m_data[i] / beta1_t / (sqrtf(v_data[i] / beta2_t) + eps);
  }

  return 0;
}
