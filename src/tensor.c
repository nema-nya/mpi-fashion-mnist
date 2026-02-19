#include "tensor.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t max(size_t a, size_t b) { return a > b ? a : b; }

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

size_t dtype_byte_count(const Dtype dtype) {
  if (dtype == DTYPE_FLOAT32) {
    return sizeof(float);
  } else if (dtype == DTYPE_UINT8) {
    return sizeof(uint8_t);
  }
}

Tensor *tensor_alloc(const Shape shape, const Dtype dtype) {
  Tensor *t = (Tensor *)malloc(sizeof(Tensor));
  if (tensor_init(t, shape, dtype))
    return NULL;
  return t;
}

int tensor_init(Tensor *t, const Shape shape, const Dtype dtype) {
  if (!t)
    return 1;
  if (shape.rank > MAX_RANK)
    return 2;
  for (size_t i = 0; i < shape.rank; ++i)
    if (shape.dims[i] == 0)
      return 3;
  t->shape = shape;
  t->size = numel(shape);
  t->dtype = dtype;
  t->data = malloc(dtype_byte_count(dtype) * t->size);

  if (!t->data) {
    return 4;
  }

  return 0;
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
    return 1;
  if (a->dtype == DTYPE_FLOAT32) {
    float *a_float_data = (float *)a->data;
    float *b_float_data = (float *)b->data;
    for (size_t i = 0; i < a->size; ++i) {
      a_float_data[i] *= b_float_data[i];
    }
  } else if (a->dtype == DTYPE_UINT8) {
    uint8_t *a_uint8_data = (uint8_t *)a->data;
    uint8_t *b_uint8_data = (uint8_t *)b->data;
    for (size_t i = 0; i < a->size; ++i) {
      a_uint8_data[i] *= b_uint8_data[i];
    }
  }
  return 0;
}

int tensor_add(Tensor *a, const Tensor *b) {
  if (!assert_tensors(a, b))
    return 1;
  if (a->dtype == DTYPE_FLOAT32) {
    float *a_float_data = (float *)a->data;
    float *b_float_data = (float *)b->data;
    for (size_t i = 0; i < a->size; ++i) {
      a_float_data[i] += b_float_data[i];
    }
  } else if (a->dtype == DTYPE_UINT8) {
    uint8_t *a_uint8_data = (uint8_t *)a->data;
    uint8_t *b_uint8_data = (uint8_t *)b->data;
    for (size_t i = 0; i < a->size; ++i) {
      a_uint8_data[i] += b_uint8_data[i];
    }
  }
  return 0;
}

int tensor_fill_float(Tensor *t, float value) {
  if (!t || !t->data)
    return 1;

  float *t_float_data = (float *)t->data;
  for (size_t i = 0; i < t->size; ++i)
    t_float_data[i] = value;
  return 0;
}

int tensor_fill_uint8(Tensor *t, uint8_t value) {
  if (!t || !t->data)
    return 1;

  uint8_t *t_uint8_data = (uint8_t *)t->data;
  for (size_t i = 0; i < t->size; ++i)
    t_uint8_data[i] = value;
  return 0;
}

int tensor_zero_float(Tensor *t) { return tensor_fill_float(t, 0.0f); }

int tensor_zero_uint8(Tensor *t) { return tensor_fill_uint8(t, 0); }

int tensor_scale_float(Tensor *t, float a) {
  if (!t || !t->data)
    return 1;

  float *t_float_data = (float *)t->data;
  for (size_t i = 0; i < t->size; ++i)
    t_float_data[i] *= a;
  return 0;
}

int tensor_scale_uint8(Tensor *t, uint8_t a) {
  if (!t || !t->data)
    return 1;

  uint8_t *t_uint_data = (uint8_t *)t->data;
  for (size_t i = 0; i < t->size; ++i)
    t_uint_data[i] *= a;
  return 0;
}

int tensor_copy(Tensor *dst, const Tensor *src) {
  if (!assert_tensors(dst, src))
    return 1;

  memcpy(dst->data, src->data, dst->size);
  return 0;
}

int tensor_fill_rand_uniform(Tensor *t, RNG *r) {
  if (!t || !t->data || !r)
    return 1;
  if (t->dtype != DTYPE_FLOAT32)
    return 2;

  float *float_data = (float *)t->data;
  for (size_t i = 0; i < t->size; ++i)
    float_data[i] = rng_uniform(r);
  return 1;
}

int tensor_fill_rand_normal(Tensor *t, RNG *r) {
  if (!t || !t->data || !r)
    return 1;

  if (t->dtype != DTYPE_FLOAT32)
    return 2;

  float *float_data = (float *)t->data;
  for (size_t i = 0; i < t->size; ++i)
    float_data[i] = rng_normal(r);
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
    return 1;
  int t_shape_size = numel(t->shape);
  int s_shape_size = numel(shape);
  if (t_shape_size != s_shape_size)
    return 2;
  t->shape = shape;
  return 0;
}

int tensor_unindex(const Shape shape, size_t ix, size_t *ixs) {
  for (size_t i = 0; i < shape.rank; ++i) {
    size_t r_i = shape.rank - i - 1;
    ixs[r_i] = ix % shape.dims[r_i];
    ix /= shape.dims[r_i];
  }
  return 0;
}

size_t tensor_byte_count(const Tensor *t) {
  return t->size * dtype_byte_count(t->dtype);
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
  void *new_buffer = (void *)malloc(tensor_byte_count(t));
  memcpy(new_buffer, t->data, tensor_byte_count(t));
  if (t->dtype == DTYPE_FLOAT32) {
    float *data = (float *)t->data;
    float *new_data = (float *)new_buffer;
    for (size_t i = 0; i < t->size; ++i) {
      tensor_unindex(t->shape, i, indicies);
      for (size_t j = 0; j < t->shape.rank; ++j) {
        permuted_indicies[j] = indicies[permutation[j]];
      }
      size_t new_i = tensor_index_array(s, permuted_indicies);
      data[new_i] = new_data[i];
    }
  } else if (t->dtype == DTYPE_UINT8) {
    uint8_t *data = (uint8_t *)t->data;
    uint8_t *new_data = (uint8_t *)new_buffer;
    for (size_t i = 0; i < t->size; ++i) {
      tensor_unindex(t->shape, i, indicies);
      for (size_t j = 0; j < t->shape.rank; ++j) {
        permuted_indicies[j] = indicies[permutation[j]];
      }
      size_t new_i = tensor_index_array(s, permuted_indicies);
      data[new_i] = new_data[i];
    }
  }
  t->shape = s;
  free(new_buffer);
  return 0;
}

int tensor_arange_float(Tensor *t) {
  float *data = (float *)t->data;
  for (size_t i = 0; i < t->size; ++i) {
    data[i] = i;
  }
  return 0;
}

int tensor_arange_uint8(Tensor *t) {
  uint8_t *data = (uint8_t *)t->data;
  for (size_t i = 0; i < t->size; ++i) {
    data[i] = i;
  }
  return 0;
}

int tensor_expand(const Tensor *src, Tensor *dest) {
  if (!src || !dest) {
    return 1;
  }

  Shape shape_dest;
  if (shape_expand(src->shape, dest->shape, &shape_dest)) {
    return 2;
  }

  if (!shape_is_equal(dest->shape, shape_dest)) {
    return 3;
  }
  size_t indicies[MAX_RANK];
  size_t new_dims = dest->shape.rank - src->shape.rank;
  for (size_t i = 0; i < dest->size; ++i) {
    tensor_unindex(shape_dest, i, indicies);
    for (size_t j = 0; j < dest->shape.rank; ++j) {
      if (j < new_dims) {
        indicies[j] = 0;
      } else {
        if (dest->shape.dims[j] > src->shape.dims[j - new_dims]) {
          indicies[j] = 0;
        }
      }
    }
    if (dest->dtype == DTYPE_FLOAT32) {
      float *dest_data = (float *)dest->data;
      float *src_data = (float *)src->data;
      for (size_t j = 0; j < dest->shape.rank; ++j) {
        indicies[j] = indicies[j + new_dims];
      }
      size_t src_i = tensor_index_array(src->shape, indicies);
      dest_data[i] = src_data[src_i];
    } else if (dest->dtype == DTYPE_UINT8) {
      uint8_t *dest_data = (uint8_t *)dest->data;
      uint8_t *src_data = (uint8_t *)src->data;
      for (size_t j = 0; j < dest->shape.rank; ++j) {
        indicies[j] = indicies[j + new_dims];
      }
      size_t src_i = tensor_index_array(src->shape, indicies);
      dest_data[i] = src_data[src_i];
    }
  }
  return 0;
}

int tensor_slice(const Tensor *src, Tensor *dest, size_t dim, size_t start,
                 size_t end) {
  if (src->shape.rank != dest->shape.rank) {
    return 1;
  }
  for (size_t i = 0; i < src->shape.rank; ++i) {
    if (i == dim) {
      if (dest->shape.dims[i] != end - start) {
        return 2;
      } else if (src->shape.dims[i] != dest->shape.dims[i]) {
        return 3;
      }
    }
  }

  if (src->dtype != dest->dtype) {
    return 4;
  }

  size_t indicies[MAX_RANK];
  if (src->dtype == DTYPE_FLOAT32) {
    float *src_data = (float *)src->data;
    float *dest_data = (float *)dest->data;
    for (size_t i = 0; i < dest->size; ++i) {
      tensor_unindex(dest->shape, i, indicies);
      indicies[dim] = indicies[dim] + start;
      size_t src_i = tensor_index_array(src->shape, indicies);
      dest_data[src_i] = src_data[i];
    }
  } else if (src->dtype == DTYPE_UINT8) {
    uint8_t *src_data = (uint8_t *)src->data;
    uint8_t *dest_data = (uint8_t *)dest->data;
    for (size_t i = 0; i < dest->size; ++i) {
      tensor_unindex(dest->shape, i, indicies);
      indicies[dim] = indicies[dim] + start;
      size_t src_i = tensor_index_array(src->shape, indicies);
      dest_data[src_i] = src_data[i];
    }
  }
  return 0;
}

bool shape_is_equal(const Shape a, const Shape b) {
  if (a.rank != b.rank) {
    return false;
  }
  for (size_t i = 0; i < a.rank; ++i) {
    if (a.dims[i] != b.dims[i]) {
      return false;
    }
  }
  return true;
}

int shape_expand(const Shape l, const Shape r, Shape *out) {
  size_t max_rank = max(l.rank, r.rank);
  Shape l_ex;
  Shape r_ex;
  l_ex.rank = max_rank;
  r_ex.rank = max_rank;
  size_t l_pad = max_rank - l.rank;
  size_t r_pad = max_rank - r.rank;
  out->rank = max_rank;
  for (size_t i = 0; i < max_rank; ++i) {
    if (i < l_pad) {
      l_ex.dims[i] = 1;
    } else {
      l_ex.dims[i] = l.dims[i - l_pad];
    }
    if (i < r_pad) {
      r_ex.dims[i] = 1;
    } else {
      r_ex.dims[i] = r.dims[i - r_pad];
    }
  }

  for (size_t i = 0; i < max_rank; ++i) {
    if ((l_ex.dims[i] != r_ex.dims[i]) &&
        (l_ex.dims[i] != 1 && r_ex.dims[i] != 1)) {
      return 1;
    }
    out->dims[i] = max(l_ex.dims[i], r_ex.dims[i]);
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

Shape shapeN(size_t rank, ...) {
  va_list args;
  va_start(args);
  Shape n;
  n.rank = rank;
  for (size_t i = 0; i < rank; ++i) {
    size_t ix = va_arg(args, int);
    n.dims[i] = ix;
  }
  return n;
}

void print_shape(const Tensor *t) {
  for (size_t i = 0; i < t->shape.rank; ++i) {
    printf("%zu ", t->shape.dims[i]);
  }
  printf("\n\r");
}

void print_tensor(const Tensor *t) {
  if (t->dtype == DTYPE_FLOAT32) {
    float *data = (float *)t->data;
    for (size_t i = 0; i < t->size; ++i) {
      printf("%f ", data[i]);
    }
  } else if (t->dtype == DTYPE_UINT8) {
    uint8_t *data = (uint8_t *)t->data;
    for (size_t i = 0; i < t->size; ++i) {
      printf("%hhu ", data[i]);
    }
  }
  printf("\n\r");
}
