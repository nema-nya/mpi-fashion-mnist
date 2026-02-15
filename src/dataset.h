#ifndef DATASET_H
#define DATASET_H

#include "tensor.h"
#include <stddef.h>
#include <stdint.h>

#define IMG_SIZE 784

typedef struct {
  size_t n;
  Tensor *y;
  Tensor *x;
} Dataset;

int dataset_load_csv(const char *path, Dataset *out);
void dataset_free(Dataset *d);

int dataset_load_bin(const char *labels_path, const char *data_path,
                     Dataset *out);

#endif
