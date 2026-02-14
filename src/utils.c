#include "utils.h"

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
