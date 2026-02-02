#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#define MAX_DIMS 8

typedef struct {
    size_t dims[MAX_DIMS];
    size_t ndims;
} Shape;

typedef struct {
    float* data;
    Shape shape;
    size_t size;
} Tensor;

int tensor_alloc(Tensor* t, size_t, const size_t* dims);
void tensor_free(Tensor* t);

size_t tensor_size(const Tensor* t);
size_t tensor_dim(const Tensor* t, size_t i);

int tensor_same_shape(const Tensor* a, const Tensor* b);

int tensor_mul(Tensor* a, const Tensor* b);

#endif