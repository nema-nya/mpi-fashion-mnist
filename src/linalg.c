#include "linalg.h"
#include "tensor.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <vcruntime.h>

int bmm(Tensor *C, const Tensor *A, const Tensor *B) {
  if (!C || !A || !B)
    return 1;
  if (C->shape.rank != 3 || A->shape.rank != 3 || B->shape.rank != 3)
    return 2;
  if (A->shape.dims[2] != B->shape.dims[1])
    return 3;
  if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32 ||
      C->dtype != DTYPE_FLOAT32)
    return 4;
  float *a_data = (float *)A->data;
  float *b_data = (float *)B->data;
  float *c_data = (float *)C->data;
  for (size_t i = 0; i < C->shape.dims[0]; ++i) {
    for (size_t j = 0; j < C->shape.dims[1]; ++j) {
      for (size_t k = 0; k < C->shape.dims[2]; ++k) {
        size_t c_index = tensor_index(C->shape, i, j, k);
        float sum = 0.0f;
        for (size_t l = 0; l < A->shape.dims[2]; ++l) {
          size_t a_index =
              tensor_index(A->shape, i % A->shape.dims[0], j % A->shape.dims[1],
                           l % A->shape.dims[2]);
          size_t b_index =
              tensor_index(B->shape, i % B->shape.dims[0], l % B->shape.dims[1],
                           k % B->shape.dims[2]);
          sum += a_data[a_index] * b_data[b_index];
          // printf("a_index %zu b_index %zu c_index %zu\n", a_index, b_index,
          // c_index);
        }
        c_data[c_index] = sum;
      }
    }
  }
  return 0;
}

int tensor_add(Tensor *a, const Tensor *b) {
  if (!a || !b) {
    return 1;
  }

  if (a->dtype != b->dtype) {
    return 2;
  }

  Shape shape_dest;
  if (shape_expand(a->shape, b->shape, &shape_dest)) {
    return 3;
  }

  if (!shape_is_equal(a->shape, shape_dest)) {
    return 4;
  }
  size_t b_indicies[MAX_RANK];
  size_t new_dims = shape_dest.rank - b->shape.rank;
  for (size_t i = 0; i < a->size; ++i) {
    tensor_unindex(shape_dest, i, b_indicies);

    for (size_t j = 0; j < shape_dest.rank; ++j) {
      if (j < new_dims) {
        b_indicies[j] = 0;
      } else {
        if (shape_dest.dims[j] > b->shape.dims[j - new_dims]) {
          b_indicies[j] = 0;
        }
      }
    }

    if (b->dtype == DTYPE_FLOAT32) {
      float *a_data = (float *)a->data;
      float *b_data = (float *)b->data;
      for (size_t j = 0; j < b->shape.rank; ++j) {
        b_indicies[j] = b_indicies[j + new_dims];
      }
      size_t src_i = tensor_index_array(b->shape, b_indicies);
      a_data[i] += b_data[src_i];
    } else if (b->dtype == DTYPE_UINT8) {
      uint8_t *a_data = (uint8_t *)a->data;
      uint8_t *b_data = (uint8_t *)b->data;
      for (size_t j = 0; j < b->shape.rank; ++j) {
        b_indicies[j] = b_indicies[j + new_dims];
      }
      size_t src_i = tensor_index_array(b->shape, b_indicies);
      a_data[i] += b_data[src_i];
    }
  }
  return 0;
}

int tensor_tanh(Tensor *a) {
  if (a == NULL) {
    return 1;
  }
  if (a->dtype != DTYPE_FLOAT32) {
    return 2;
  }
  float *a_data = (float *)a->data;
  for (size_t i = 0; i < a->size; ++i) {
    a_data[i] = tanh(a_data[i]);
  }

  return 0;
}

int tensor_argmax(const Tensor *a, Tensor *out) {
  if (!a || !out) {
    return 1;
  }

  if (a->shape.rank == 0) {
    return 2;
  }

  if (a->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_UINT8) {
    return 3;
  }

  Shape s;
  s.rank = a->shape.rank - 1;
  for (size_t i = 0; i < s.rank; ++i) {
    s.dims[i] = a->shape.dims[i];
  }

  if (!shape_is_equal(s, out->shape)) {
    return 3;
  }

  size_t indicies[MAX_RANK];
  float *a_data = (float *)a->data;
  uint8_t *out_data = (uint8_t *)out->data;
  for (size_t i = 0; i < out->size; ++i) {
    tensor_unindex(out->shape, i, indicies);
    indicies[s.rank] = 0;
    size_t new_i = tensor_index_array(a->shape, indicies);
    size_t argmax = 0;
    for (size_t j = 1; j < a->shape.dims[s.rank]; ++j) {
      size_t a_i = new_i + j;
      if (a_data[a_i] > a_data[new_i + argmax]) {
        argmax = j;
      }
    }
    out_data[i] = argmax;
  }

  return 0;
}

int accuracy(const Tensor *a, const Tensor *b, float *acc) {
  if (!a || !b) {
    return 1;
  }

  if (a->dtype != DTYPE_UINT8 || b->dtype != DTYPE_UINT8) {
    return 2;
  }

  if (!shape_is_equal(a->shape, b->shape)) {
    return 3;
  }

  uint8_t *a_data = (uint8_t *)a->data;
  uint8_t *b_data = (uint8_t *)b->data;
  size_t matches = 0;
  for (size_t i = 0; i < a->size; ++i) {
    if (a_data[i] == b_data[i]) {
      ++matches;
    }
  }

  *acc = (float)(matches) / a->size;

  return 0;
}

int cross_entropy(const Tensor *y_, const Tensor *y, float *loss) {
  if (!y_ || !y) {
    return 1;
  }

  if (y_->dtype != DTYPE_FLOAT32 || y->dtype != DTYPE_UINT8) {
    return 2;
  }

  Shape s;
  s.rank = y_->shape.rank - 1;
  for (size_t i = 0; i < s.rank; ++i) {
    s.dims[i] = y_->shape.dims[i];
  }

  if (!shape_is_equal(s, y->shape)) {
    return 3;
  }

  float *y_data_ = (float *)y_->data;
  uint8_t *y_data = (uint8_t *)y->data;
  size_t indicies[MAX_RANK];
  *loss = 0.0;
  for (size_t i = 0; i < y->size; ++i) {
    tensor_unindex(y->shape, i, indicies);
    indicies[s.rank] = 0;
    size_t new_i = tensor_index_array(y_->shape, indicies);
    float denom = 0.0;
    for (size_t j = 0; j < y_->shape.dims[s.rank]; ++j) {
      size_t y_i_ = new_i + j;
      denom += exp(y_data_[y_i_]);
    }
    float nll = -log(exp(y_data_[new_i + y_data[i]]) / denom);
    *loss += nll;
  }

  *loss /= y->size;

  return 0;
}
