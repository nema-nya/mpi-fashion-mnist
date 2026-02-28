#ifndef LINALG_H
#define LINALG_H

#include <stddef.h>

#include "tensor.h"

int bmm(Tensor *C, const Tensor *A, const Tensor *B);

int tensor_add(Tensor *a, const Tensor *b);

int tensor_tanh(Tensor *a);

int tensor_argmax(const Tensor *a, Tensor *out);

int accuracy(const Tensor *a, const Tensor *b, float *acc);

int cross_entropy(const Tensor *y_, const Tensor *y, float *loss);

#endif
