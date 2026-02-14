#include "tensor.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t numel(const Shape shape) {
  size_t p = 1;
  for (size_t i = 0; i < shape.rank; ++i)
    p *= shape.dims[i];
  return p;
}

static int assert_tensors(const Tensor *a, const Tensor *b) {
  if (!a || !b)
    return 0;
  if (!a->data || !b->data)
    return 0;
  if (!tensor_same_shape(a, b))
    return 0;
  if (a->size != b->size)
    return 0;
  return 1;
}

int tensor_alloc(Tensor *t, const Shape shape) {
  if (!t)
    return 0;
  if (shape.rank == 0 || shape.rank > MAX_RANK)
    return 0;
  for (size_t i = 0; i < shape.rank; ++i)
    if (shape.dims[i] == 0)
      return 0;

  free(t->data);

  t->data = NULL;
  t->shape = shape;
  t->size = numel(shape);
  t->data = (float *)malloc(t->size * sizeof(float));

  if (!t->data) {
    t->size = 0;
    t->shape.rank = 0;
    return 0;
  }

  return 1;
}

void tensor_free(Tensor *t) {
  if (!t)
    return;
  free(t->data);
  t->data = NULL;
  t->shape.rank = 0;
  t->size = 0;
}

size_t tensor_size(const Tensor *t) { return t ? t->size : 0; }

size_t tensor_dim(const Tensor *t, size_t i) {
  if (!t)
    return 0;
  if (i >= t->shape.rank)
    return 0;
  return t->shape.dims[i];
}

int tensor_same_shape(const Tensor *a, const Tensor *b) {
  if (!a || !b)
    return 0;
  if (a->shape.rank != b->shape.rank)
    return 0;
  for (size_t i = 0; i < a->shape.rank; ++i) {
    if (a->shape.dims[i] != b->shape.dims[i])
      return 0;
  }
  return 1;
}

int tensor_mul(Tensor *a, const Tensor *b) {
  if (!assert_tensors(a, b))
    return 0;
  for (size_t i = 0; i < a->size; ++i)
    a->data[i] *= b->data[i];
  return 1;
}

int tensor_add(Tensor *a, const Tensor *b) {
  if (!assert_tensors(a, b))
    return 0;
  for (size_t i = 0; i < a->size; ++i)
    a->data[i] += b->data[i];
  return 1;
}

int tensor_fill(Tensor *t, float value) {
  if (!t || !t->data)
    return 0;
  for (size_t i = 0; i < t->size; ++i)
    t->data[i] = value;
  return 1;
}

int tensor_zero(Tensor *t) { return tensor_fill(t, 0.0f); }

int tensor_scale(Tensor *t, float a) {
  if (!t || !t->data)
    return 0;
  for (size_t i = 0; i < t->size; ++i)
    t->data[i] *= a;
  return 1;
}

int tensor_copy(Tensor *dst, const Tensor *src) {
  if (!assert_tensors(dst, src))
    return 0;
  for (size_t i = 0; i < src->size; ++i)
    dst->data[i] = src->data[i];
  return 1;
}

int tensor_clone(Tensor *dst, const Tensor *src) {
  if (!dst || !src)
    return 0;
  if (!src->data)
    return 0;
  if (!tensor_alloc(dst, src->shape))
    return 0;

  for (size_t i = 0; i < src->size; ++i)
    dst->data[i] = src->data[i];
  return 1;
}

int tensor_axpy(Tensor *y, float a, const Tensor *x) {
  if (!assert_tensors(y, x))
    return 0;

  for (size_t i = 0; i < y->size; ++i)
    y->data[i] += a * x->data[i];
  return 1;
}

float tensor_get(const Tensor *t, size_t index) {
  if (!t || !t->data || index >= t->size)
    return 0.0f;
  return t->data[index];
}
int tensor_fill_rand_uniform(Tensor *t, RNG *r) {
  if (!t || !t->data || !r)
    return 0;
  for (size_t i = 0; i < t->size; ++i)
    t->data[i] = rng_uniform(r);
  return 1;
}
int tensor_fill_rand_normal(Tensor *t, RNG *r) {
  if (!t || !t->data || !r)
    return 0;
  for (size_t i = 0; i < t->size; ++i)
    t->data[i] = rng_normal(r);
  return 1;
}

int tensor_index(const Shape shape, ...) {
  va_list args;
  va_start(args);
  int out = 0;
  for (size_t i = 0; i < shape.rank; ++i) {
    size_t ix = va_arg(args, int);
    out *= shape.dims[i];
    out += ix;
  }
  return out;
}

int tensor_index_array(const Shape shape, size_t *indicies) {
  int out = 0;
  for (size_t i = 0; i < shape.rank; ++i) {
    out *= shape.dims[i];
    out += indicies[i];
  }
  return out;
}

int reshape(Tensor *t, const Shape shape) {
  if (!t)
    return 0;
  int t_shape_size = numel(t->shape);
  int s_shape_size = numel(shape);
  if (t_shape_size != s_shape_size)
    return 0;
  t->shape = shape;
  return 1;
}

int tensor_unindex(const Shape shape, size_t ix, size_t *ixs) {
  for (size_t i = 0; i < shape.rank; ++i) {
    size_t r_i = shape.rank - i - 1;
    ixs[r_i] = ix % shape.dims[r_i];
    ix /= shape.dims[r_i];
  }
  return 0;
}

int permute(Tensor *t, ...) {
  // TODO: implement without buffer.
  if (!t)
    return 0;
  va_list args;
  va_start(args);
  size_t permutation[MAX_RANK];
  for (size_t i = 0; i < t->shape.rank; ++i) {
    int ix = va_arg(args, int);
    permutation[i] = ix;
  }
  Shape s;
  s.rank = t->shape.rank;
  for (size_t i = 0; i < s.rank; ++i) {
    s.dims[i] = t->shape.dims[permutation[i]];
  }
  size_t indicies[MAX_RANK];
  size_t permuted_indicies[MAX_RANK];
  float *new_buffer = (float *)malloc(t->size * sizeof(float));
  memcpy(new_buffer, t->data, t->size * sizeof(float));
  for (size_t i = 0; i < t->size; ++i) {
    tensor_unindex(t->shape, i, indicies);
    for (size_t j = 0; j < t->shape.rank; ++j) {
      permuted_indicies[j] = indicies[permutation[j]];
    }
    size_t new_i = tensor_index_array(s, permuted_indicies);
    t->data[new_i] = new_buffer[i];
  }
  t->shape = s;
  free(new_buffer);
  return 0;
}

int tensor_arange(Tensor *t) {
  for (size_t i = 0; i < t->size; ++i) {
    t->data[i] = i;
  }
  return 0;
}

Shape shape1(size_t d0) { return (Shape){.dims = {d0}, .rank = 1}; }

Shape shape2(size_t d0, size_t d1) {
  return (Shape){.dims = {d0, d1}, .rank = 2};
}

Shape shape3(size_t d0, size_t d1, size_t d2) {
  return (Shape){.dims = {d0, d1, d2}, .rank = 3};
}
