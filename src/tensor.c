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

int tensor_expand(const Tensor *src, Tensor *dest) {
  if (!src || !dest) {
    return 1;
  }

  Shape shape_dest;
  if (shape_expand(src->shape, dest->shape, &shape_dest)) {
    return 2;
  }

  if (!shape_is_equal(dest->shape, shape_dest)) {
    for (size_t i = 0; i < dest->shape.rank; ++i) {
      printf("%zu ", dest->shape.dims[i]);
    }
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
    for (size_t z = 0; z < dest->shape.rank; ++z) {
      printf("%zu ", indicies[z]);
    }
    printf("\n\r");
    for (size_t j = 0; j < dest->shape.rank; ++j) {
      indicies[j] = indicies[j + new_dims];
    }
    size_t src_i = tensor_index_array(src->shape, indicies);
    printf("i am setting dest data [%zu] to %.2f from src_i[%zu]\r\n", i,
           src->data[src_i], src_i);
    dest->data[i] = src->data[src_i];
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