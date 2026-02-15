#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
  Tensor *new_x = tensor_alloc(shapeN(3, 100, 1, 784), DTYPE_FLOAT32);
  Tensor *new_y = tensor_alloc(shapeN(1, 100), DTYPE_UINT8);
  tensor_slice(d.x, new_x, 0, 0, 100);
  tensor_slice(d.y, new_y, 0, 0, 100);
  free(d.x);
  free(d.y);
  d.x = new_x;
  d.y = new_y;
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

  // tensor_arange_float(&t2);
  // print_tensor(&t2);
  // printf("\n\r");
  // int val = tensor_expand(&t2, &t1);
  // printf("%d\n\r", val);
  // print_tensor(&t1);

  float *layer1_weight_data = (float *)read_all("data/layer1-weight.bin", NULL);
  float *layer1_bias_data = (float *)read_all("data/layer1-bias.bin", NULL);
  float *layer2_weight_data = (float *)read_all("data/layer2-weight.bin", NULL);
  float *layer2_bias_data = (float *)read_all("data/layer2-bias.bin", NULL);

  Tensor *layer1_weight = tensor_alloc(shapeN(2, 784, 256), DTYPE_FLOAT32);
  Tensor *layer1_bias = tensor_alloc(shapeN(1, 256), DTYPE_FLOAT32);
  Tensor *layer2_weight = tensor_alloc(shapeN(2, 256, 10), DTYPE_FLOAT32);
  Tensor *layer2_bias = tensor_alloc(shapeN(1, 10), DTYPE_FLOAT32);

  free(layer1_weight->data);
  free(layer1_bias->data);
  free(layer2_weight->data);
  free(layer2_bias->data);

  layer1_weight->data = (void *)layer1_weight_data;
  layer1_bias->data = (void *)layer1_bias_data;
  layer2_weight->data = (void *)layer2_weight_data;
  layer2_bias->data = (void *)layer2_bias_data;
  printf("%.2f", layer1_weight_data[0]);
  int rs_ret1 = reshape(layer1_weight, shapeN(3, 1, 784, 256));
  printf("rs_ret1 - %d\r\n", rs_ret1);
  // print_tensor(layer1_bias);
  int rs_ret2 = reshape(d.x, shapeN(3, 100, 1, 784));
  printf("rs_ret2 - %d\r\n", rs_ret2);
  // 1 forward pass
  Tensor *hidden_1 = tensor_alloc(shapeN(3, 100, 1, 256), DTYPE_FLOAT32);
  int ret = bmm(hidden_1, d.x, layer1_weight);
  printf("ret - %d", ret);
  print_tensor(hidden_1); 
  tensor_free(&t1);
  tensor_free(&t2);
  dataset_free(&d);
  return 0;
}
