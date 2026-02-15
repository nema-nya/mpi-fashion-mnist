#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "dataset.h"
#include "linalg.h"
#include "tensor.h"
#include "utils.h"

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
  Shape s_2 = shape3(3, 1, 5);
  Shape s_1 = shapeN(4, 1, 3, 2, 5);
  if (tensor_init(&t1, s_1, DTYPE_FLOAT32)) {
    printf("failed to allocate tensors 1\n");
    tensor_free(&t1);
    tensor_free(&t2);
    dataset_free(&d);
    return 1;
  }
  if (tensor_init(&t2, s_2, DTYPE_FLOAT32)) {
    printf("failed to allocate tensors 2\n");
    tensor_free(&t1);
    tensor_free(&t2);
    dataset_free(&d);
    return 1;
  }

  tensor_arange_float(&t2);

  print_tensor(&t2);
  printf("\n\r");
  int val = tensor_expand(&t2, &t1);
  printf("%d\n\r", val);
  print_tensor(&t1);

  // float *layer1_weight = (float *)read_all("data/layer1-weight.bin", NULL);
  // float *layer1_bias = (float *)read_all("data/layer1-bias.bin", NULL);
  // float *layer2_weight = (float *)read_all("data/layer2-weight.bin", NULL);
  // float *layer2_bias = (float *)read_all("data/layer1-bias.bin", NULL);

  tensor_free(&t1);
  tensor_free(&t2);

  dataset_free(&d);
  return 0;
}
