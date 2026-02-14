#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void *read_all(const char *path, size_t *n) {
  FILE *file = NULL;
  void *buffer = NULL;

  file = fopen(path, "rb");

  if (file == NULL) {
    goto on_error;
  }

  if (fseek(file, 0, SEEK_END)) {
    goto on_error;
  }

  size_t data_size = ftell(file);

  if (fseek(file, 0, SEEK_SET)) {
    goto on_error;
  }

  buffer = malloc(data_size);
  if (fread(buffer, 1, data_size, file) != data_size) {
    goto on_error;
  }

  fclose(file);
  file = NULL;
  if (n != NULL) {
    *n = data_size;
  }
  return buffer;

on_error:
  free(buffer);
  buffer = NULL;
  if (file) {
    fclose(file);
    file = NULL;
  }
  return NULL;
}

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

  out->n = n;
  out->x = data_buffer;
  out->y = labels_buffer;
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

  out->n = n;
  out->y = labels;
  out->x = images;
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
