#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "dataset.h"
#include "linalg.h"
#include "tensor.h"

bool verify_endianness() {
  uint16_t dummy = 0x0100;
  uint8_t *dummy_ptr = (uint8_t *)(&dummy);
  return dummy_ptr[0] == 0 && dummy_ptr[1] == 1;
}

int main(void) {
  assert(("Your system is big-endian", verify_endianness()));
  Dataset d;
  if (dataset_load_bin("data/fashion-mnist_train-labels.bin",
                       "data/fashion-mnist_train-data.bin", &d)) {
    printf("failed to load dataset\n");
    return 1;
  }

  printf("loaded %zu samples\n", d.n);

  Tensor t1 = {0};
  Tensor t2 = {0};
  Tensor t3 = {0};
  Shape s = shape3(5, 3, 2);
  if (!tensor_alloc(&t1, s) || !tensor_alloc(&t2, s) || !tensor_alloc(&t3, s)) {
    printf("failed to allocate tensors\n");
    tensor_free(&t1);
    tensor_free(&t2);
    dataset_free(&d);
    return 1;
  }

  // if (!tensor_fill(&t1, 1.5f) || !tensor_fill(&t2, 1.5f) ||
  // !tensor_zero(&t3)) {
  //   printf("tensor operation failed\n");
  //   tensor_free(&t1);
  //   tensor_free(&t2);
  //   dataset_free(&d);
  //   return 1;
  // }
  if (!bmm(&t3, &t1, &t2)) {
    printf("failed\n");
  }
  // RNG rng = {0};
  // rng.state = 67;
  // tensor_fill_rand_normal(&t1, &rng);

  // for (size_t i = 0; i < t3.size; ++i) {
  //   printf("%f\n", t3.data[i]);
  // }

  // Shape s = shape3(3, 3, 3);

  // for (size_t i = 0; i < t1.size; ++i) {
  //   printf("%f ", t1.data[i]);
  // }
  tensor_arange(&t1);
  permute(&t1, 2, 1, 0);
  for (size_t i = 0; i < t1.size; ++i) {
    printf("%f ", t1.data[i]);
  }
  printf("\r\n");
  tensor_free(&t1);
  tensor_free(&t2);

  dataset_free(&d);
  return 0;
}
