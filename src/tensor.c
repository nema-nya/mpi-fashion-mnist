#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t numel(const Shape shape) {
    size_t p = 1;
    for (size_t i = 0; i < shape.ndims; ++i) p *= shape.dims[i];
    return p;
}

static int assert_tensor(const Tensor* a, const Tensor* b) {
    if (!a || !b) return 0;
    if (!a->data || !b->data) return 0;
    if (!tensor_same_shape(a, b)) return 0;
    if (a->size != b->size) return 0;
    return 1;
}

int tensor_alloc(Tensor* t, const Shape shape) {
    if (!t) return 0;
    if (shape.ndims == 0 || shape.ndims > MAX_DIMS) return 0;
    for (size_t i = 0; i < shape.ndims; ++i)
        if (shape.dims[i] == 0) return 0;

    t->data = NULL;
    t->shape = shape;
    t->size = numel(shape);
    t->data = (float*)malloc(t->size * sizeof(float));

    if (!t->data) {
        t->size = 0;
        t->shape.ndims = 0;
        return 0;
    }

    return 1;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    free(t->data);
    t->data = NULL;
    t->shape.ndims = 0;
    t->size = 0;
}

size_t tensor_size(const Tensor* t) { return t ? t->size : 0; }

size_t tensor_dim(const Tensor* t, size_t i) {
    if (!t) return 0;
    if (i >= t->shape.ndims) return 0;
    return t->shape.dims[i];
}

int tensor_same_shape(const Tensor* a, const Tensor* b) {
    if (!a || !b) return 0;
    if (a->shape.ndims != b->shape.ndims) return 0;
    for (size_t i = 0; i < a->shape.ndims; ++i) {
        if (a->shape.dims[i] != b->shape.dims[i]) return 0;
    }
    return 1;
}

int tensor_mul(Tensor* a, const Tensor* b) {
    if (!assert_tensor(a, b)) return 0;
    for (size_t i = 0; i < a->size; ++i) a->data[i] *= b->data[i];
    return 1;
}

int tensor_add(Tensor* a, const Tensor* b) {
    if (!assert_tensor(a, b)) return 0;
    for (size_t i = 0; i < a->size; ++i) a->data[i] += b->data[i];
    return 1;
}

int tensor_fill(Tensor* t, float value) {
    if (!t || !t->data) return 0;
    for (size_t i = 0; i < t->size; ++i) t->data[i] = value;
    return 1;
}

int tensor_zero(Tensor* t) { return tensor_fill(t, 0.0f); }

int tensor_scale(Tensor* t, float a) {
    if (!t || !t->data) return 0;
    for (size_t i = 0; i < t->size; ++i) t->data[i] *= a;
    return 1;
}
