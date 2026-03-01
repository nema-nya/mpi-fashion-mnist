#ifndef LINALG_H
#define LINALG_H

#include "tensor.h"
#include <stddef.h>

int bmm(Tensor *C, const Tensor *A, const Tensor *B);

int tensor_add(Tensor *a, const Tensor *b);

int tensor_tanh(Tensor *a);

int tensor_argmax(const Tensor *a, Tensor *out);

int accuracy(const Tensor *a, const Tensor *b, float *acc);

int cross_entropy(const Tensor *y_, const Tensor *y, float *loss);

int cross_entropy_backward(const Tensor *y_, const Tensor *y, Tensor *y_grad_);

int tensor_tanh_backward(const Tensor *a, Tensor *a_grad);

int tensor_bcast_grad(const Tensor *y_grad, Tensor *x_grad);

int tensor_add_backward(const Tensor *ab_grad, Tensor *a_grad, Tensor *b_grad);

#endif
