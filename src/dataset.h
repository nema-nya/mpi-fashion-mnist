#ifndef DATASET_H
#define DATASET_H

#include <stddef.h>
#include <stdint.h>

#define IMG_SIZE 784

typedef struct {
    size_t n;
    uint8_t* y;
    float* x;
} Dataset;

int dataset_load_csv(const char* path, Dataset* out);
void dataset_free(Dataset* d);

int dataset_load_bin(const char* labels_path, const char* data_path , Dataset* out);

#endif
