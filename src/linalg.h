#ifndef LINALG_H
#define LINALG_H

#include <stddef.h>

#include "tensor.h"

int bmm(Tensor *C, const Tensor *A, const Tensor *B);

int tensor_add(Tensor* a, const Tensor* b);

#endif
