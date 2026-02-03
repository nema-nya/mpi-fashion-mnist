#include "tensor.h"

#include <stdlib.h>

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

    free(t->data);

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

int tensor_copy(Tensor* dst, const Tensor* src) {
    if (!assert_tensor(dst, src)) return 0;
    for (size_t i = 0; i < src->size; ++i) dst->data[i] = src->data[i];
    return 1;
}

int tensor_clone(Tensor* dst, const Tensor* src) {
    if (!dst || !src) return 0;
    if (!src->data) return 0;
    if (!tensor_alloc(dst, src->shape)) return 0;

    for (size_t i = 0; i < src->size; ++i) dst->data[i] = src->data[i];
    return 1;
}

int tensor_axpy(Tensor* y, float a, const Tensor* x) {
    if (!assert_tensor(y, x)) return 0;

    for (size_t i = 0; i < y->size; ++i) y->data[i] += a * x->data[i];
    return 1;
}

float tensor_get(const Tensor* t, size_t index) {
    if (!t || !t->data || index >= t->size) return 0.0f;
    return t->data[index];
}
int tensor_fill_rand_uniform(Tensor* t, RNG* r) {
    if (!t || !t->data || !r) return 0;
    for (size_t i = 0; i < t->size; ++i) t->data[i] = rng_uniform(r);
    return 1;
}
int tensor_fill_rand_normal(Tensor* t, RNG* r) {
    if (!t || !t->data || !r) return 0;
    for (size_t i = 0; i < t->size; ++i) t->data[i] = rng_normal(r);
    return 1;
}

Shape shape1(size_t d0) { return (Shape){.dims = {d0}, .ndims = 1}; }

Shape shape2(size_t d0, size_t d1) {
    return (Shape){.dims = {d0, d1}, .ndims = 2};
}

Shape shape3(size_t d0, size_t d1, size_t d2) {
    return (Shape){.dims = {d0, d1, d2}, .ndims = 3};
}
