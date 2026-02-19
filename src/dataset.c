#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tensor.h"
#include "utils.h"

int dataset_load_bin(const char *labels_path, const char *data_path,
                     Dataset *out) {
  int ok = 0;
  uint8_t *labels_buffer = NULL;
  float *data_buffer = NULL;
  size_t n = 0;
  labels_buffer = (uint8_t *)(read_all(labels_path, &n));

  if (labels_buffer == NULL) {
    ok = 1;
    goto cleanup;
  }

  data_buffer = (float *)(read_all(data_path, NULL));
  if (data_buffer == NULL) {
    ok = 2;
    goto cleanup;
  }

  Shape x_shape = shapeN(2, n, IMG_SIZE);
  Shape y_shape = shapeN(1, n);

  Tensor *x = tensor_alloc(x_shape, DTYPE_FLOAT32);
  Tensor *y = tensor_alloc(y_shape, DTYPE_UINT8);

  x->data = data_buffer;
  y->data = labels_buffer;

  out->n = n;
  out->x = x;
  out->y = y;
  data_buffer = NULL;
  labels_buffer = NULL;

cleanup:
  free(labels_buffer);
  free(data_buffer);
  return ok;
}

static void trim_newline(char *s) {
  size_t n = strcspn(s, "\r\n");
  s[n] = '\0';
}

int dataset_load_csv(const char *path, Dataset *out) {
  FILE *fp = fopen(path, "r");

  size_t capacity = 1024;
  size_t n = 0;
  size_t buffer_size = 1024 * 16;

  Tensor *label_tensor = tensor_alloc(shapeN(1, 60000), DTYPE_UINT8);
  Tensor *images_tensor = tensor_alloc(shapeN(3, 784, 1, 60000), DTYPE_FLOAT32);

  uint8_t *labels = NULL;
  float *images = NULL;
  char buffer[buffer_size];

  int ok = 0;

  if (!fp)
    goto cleanup;

  labels = (uint8_t *)malloc(capacity * sizeof(uint8_t));
  images = (float *)malloc(capacity * IMG_SIZE * sizeof(float));
  if (!labels || !images)
    goto cleanup;

  if (!fgets(buffer, (int)buffer_size, fp))
    goto cleanup;

  while (fgets(buffer, (int)buffer_size, fp)) {
    trim_newline(buffer);

    if (n == capacity) {
      capacity *= 2;
      uint8_t *new_labels =
          (uint8_t *)realloc(labels, capacity * sizeof(uint8_t));
      float *new_images =
          (float *)realloc(images, capacity * IMG_SIZE * sizeof(float));
      if (!new_labels || !new_images) {
        if (new_labels)
          labels = new_labels;
        if (new_images)
          images = new_images;
        goto cleanup;
      }
      labels = new_labels;
      images = new_images;
    }

    char *p = buffer;
    char *endptr = NULL;

    long lab = strtol(p, &endptr, 10);
    if (endptr == p || lab < 0 || lab > 9)
      break;
    labels[n] = (uint8_t)lab;

    p = endptr;
    if (*p == ',')
      p++;
    float *img = images + n * IMG_SIZE;
    for (int j = 0; j < IMG_SIZE; ++j) {
      long v = strtol(p, &endptr, 10);
      if (endptr == p || v < 0 || v > 255)
        goto cleanup;
      img[j] = (float)v / 255.0f;
      p = endptr;
      if (*p == ',')
        p++;
    }
    n++;
  }

  label_tensor->data = labels;
  images_tensor->data = images;

  out->n = n;
  out->y = label_tensor;
  out->x = images_tensor;
  labels = NULL;
  images = NULL;
  ok = 1;

cleanup:
  if (fp)
    fclose(fp);
  free(labels);
  free(images);
  return ok;
}

void dataset_free(Dataset *d) {
  free(d->y);
  free(d->x);
  d->y = NULL;
  d->x = NULL;
  d->n = 0;
}
