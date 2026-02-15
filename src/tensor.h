#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#include "rng.h"

#define MAX_RANK 8

typedef struct {
  size_t dims[MAX_RANK];
  size_t rank;
} Shape;

typedef enum { DTYPE_FLOAT32 = 0, DTYPE_UINT8 } Dtype;

typedef struct {
  void *data;
  Shape shape;
  size_t size;
  Dtype dtype;
} Tensor;

Tensor *tensor_alloc(const Shape shape, const Dtype dtype);

int tensor_init(Tensor *t, const Shape shape, const Dtype dtype);

void tensor_free(Tensor *t);

size_t tensor_size(const Tensor *t);
size_t tensor_dim(const Tensor *t, size_t i);

int tensor_same_shape(const Tensor *a, const Tensor *b);

int tensor_mul(Tensor *a, const Tensor *b);

int tensor_add(Tensor *a, const Tensor *b);

int tensor_fill_float(Tensor *t, float value);

int tensor_fill_uint8(Tensor *t, uint8_t value);

int tensor_zero_float(Tensor *t);

int tensor_zero_uint8(Tensor *t);

int tensor_scale_float(Tensor *t, float a);

int tensor_scale_uint8(Tensor *t, uint8_t a);

int tensor_fill_rand_uniform(Tensor *t, RNG *r);

int tensor_fill_rand_normal(Tensor *t, RNG *r);

int tensor_copy(Tensor *dst, const Tensor *src);

int tensor_index(const Shape shape, ...);

int tensor_index_array(const Shape shape, size_t *indicies);

int tensor_unindex(const Shape shape, size_t ix, size_t *ixs);

int reshape(Tensor *t, const Shape shape);

int permute(Tensor *t, ...);

int tensor_arange_float(Tensor *t);

int tensor_arange_uint8(Tensor *t);

size_t tensor_byte_count(const Tensor *t);

size_t dtype_byte_count(const Dtype dtype);

int shape_expand(const Shape l, const Shape r, Shape *out);

bool shape_is_equal(const Shape a, const Shape b);

int tensor_expand(const Tensor *src, Tensor *dest);

int tensor_slice(const Tensor* src, Tensor* dest , size_t dim, size_t start, size_t end);

Shape shape1(size_t d0);

Shape shape2(size_t d0, size_t d1);

Shape shape3(size_t d0, size_t d1, size_t d2);

Shape shapeN(size_t rank, ...);

void print_shape(const Tensor *t);

void print_tensor(const Tensor *t);

#endif
