#include "linalg.h"

#include <stddef.h>

int bmm(Tensor *C, const Tensor *A, const Tensor *B) {
  if (!C || !A || !B)
    return 0;
  if (C->shape.rank != 3 || A->shape.rank != 3 || B->shape.rank != 3)
    return 0;
  if (A->shape.dims[2] != B->shape.dims[1])
    return 0;

  for (size_t i = 0; i < C->shape.dims[0]; ++i) {
    for (size_t j = 0; j < C->shape.dims[1]; ++j) {
      for (size_t k = 0; k < C->shape.dims[2]; ++k) {
        size_t c_index = tensor_index(C->shape, i, j, k);
        float sum = 0.0f;
        for (size_t l = 0; l < A->shape.dims[1]; ++l) {
          size_t a_index = tensor_index(A->shape, l, j, i);
          size_t b_index = tensor_index(B->shape, k, l, i);
          sum += A->data[a_index] * B->data[b_index];
        }
        C->data[c_index] = sum;
      }
    }
  }
  return 1;
}
