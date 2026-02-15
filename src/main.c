#include <assert.h>
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
  Tensor t3 = {0};
  // Shape s = shape3(5, 3, 2);
  // Shape s2 = shape3(2, 2, 3);
  Shape s_2 = shape3(3,1,5);
  Shape s_1 = shapeN(4, 1,3 ,2, 5);
  if (!tensor_alloc(&t1, s_1)) {
    printf("failed to allocate tensors 1\n");
    tensor_free(&t1);
    tensor_free(&t2);
    dataset_free(&d);
    return 1;
  } 
  if (!tensor_alloc(&t2, s_2)) {
    printf("failed to allocate tensors 2\n");
    tensor_free(&t1);
    tensor_free(&t2);
    dataset_free(&d);
    return 1;
  }

  // tensor_arange(&t1);
  tensor_arange(&t2);
  // Shape out;
  // shape_expand(s_1, s_2, &out);

  printf("going to expanding\r\n");
  // t2 is 3, t1 is 2, 3
  // we expand the t1.
  for (size_t i = 0; i < t2.size; ++i) {
    printf("i - %zu value - %.0f\r\n", i, t2.data[i]);
  }
  printf("\r\n");
  int val = tensor_expand(&t2, &t1);
  printf("expanding %d\r\n", val);
  for (size_t i = 0; i < t1.size; ++i) {
    printf("%.0f ", t1.data[i]);
  }
  printf("\r\n");

  // float *layer1_weight = (float *)read_all("data/layer1-weight.bin", NULL);
  // float *layer1_bias = (float *)read_all("data/layer1-bias.bin", NULL);
  // float *layer2_weight = (float *)read_all("data/layer2-weight.bin", NULL);
  // float *layer2_bias = (float *)read_all("data/layer1-bias.bin", NULL);

  // if (!tensor_fill(&t1, 1.5f) || !tensor_fill(&t2, 1.5f) ||
  // !tensor_zero(&t3)) {
  //   printf("tensor operation failed\n");
  //   tensor_free(&t1);
  //   tensor_free(&t2);
  //   dataset_free(&d);
  //   return 1;
  // }
  // if (!bmm(&t3, &t1, &t2)) {
  //   printf("failed\n");
  // }
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
  // tensor_arange(&t1);
  // permute(&t1, 2, 1, 0);
  // for (size_t i = 0; i < t1.size; ++i) {
  //   printf("%f ", t1.data[i]);
  // }
  // printf("\r\n");

  tensor_free(&t1);
  tensor_free(&t2);

  dataset_free(&d);
  return 0;
}
