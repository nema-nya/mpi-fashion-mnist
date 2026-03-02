#ifndef OPTIM_H
#define OPTIM_H

#include <linalg.h>
#include <tensor.h>
#include <utils.h>

int adam_step(float lr, float beta1, float beta2, float eps, size_t t,
              const Tensor *grad, Tensor *param, Tensor *m, Tensor *v);

#endif
