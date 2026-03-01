#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  if (dataset_load_bin("data/train-labels.bin", "data/train-data.bin", &d)) {
    printf("failed to load dataset\n");
    return 1;
  }
  Tensor *new_x = tensor_alloc(shapeN(3, 100, 1, 784), DTYPE_FLOAT32);
  Tensor *new_y = tensor_alloc(shapeN(1, 100), DTYPE_UINT8);
  // print_tensor(d.x);

  reshape(d.x, shapeN(3, 60000, 1, 784));
  // print_shape(d.x);
  int ts_ret1 = tensor_slice(d.x, new_x, 0, 0, 100);
  int ts_ret2 = tensor_slice(d.y, new_y, 0, 0, 100);
  // print_tensor(new_x);
  // printf("ts1 %d ts2 %d\n", ts_ret1, ts_ret2);
  free(d.x);
  free(d.y);
  d.x = new_x;
  d.y = new_y;
  // print_tensor(d.x);
  // printf("loaded %zu samples\n", d.n);

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
  // printf("%.2f", layer1_weight_data[0]);
  int rs_ret1 = reshape(layer1_weight, shapeN(3, 1, 784, 256));
  // printf("rs_ret1 - %d\r\n", rs_ret1);
  print_tensor(layer1_weight);
  print_tensor(layer1_bias);
  print_tensor(layer2_weight);
  print_tensor(layer2_bias);
  int rs_ret2 = reshape(d.x, shapeN(3, 100, 1, 784));
  // printf("rs_ret2 - %d\r\n", rs_ret2);
  // 1 forward pass
  Tensor *hidden_1 = tensor_alloc(shapeN(3, 100, 1, 256), DTYPE_FLOAT32);
  // print_tensor(hidden_1);
  // print_tensor(d.x);
  int ret = bmm(hidden_1, d.x, layer1_weight);
  ret = tensor_add(hidden_1, layer1_bias);
  printf("ret - %d\n", ret);

  print_tensor(hidden_1);

  tensor_tanh(hidden_1);
  print_tensor(hidden_1);
  printf("=====================\n");
  Tensor *hidden_2 = tensor_alloc(shapeN(3, 100, 1, 10), DTYPE_FLOAT32);
  reshape(layer2_weight, shapeN(3, 1, 256, 10));
  // print_tensor(hidden_1);
  // print_tensor(d.x);
  ret = bmm(hidden_2, hidden_1, layer2_weight);
  printf("ret - %d\n", ret);
  print_tensor(hidden_2);
  ret = tensor_add(hidden_2, layer2_bias);
  printf("ret - %d\n", ret);
  print_tensor(hidden_2);
  printf("=====================\n");
  Tensor *argmax_out = tensor_alloc(shapeN(2, 100, 1), DTYPE_UINT8);
  int arg_ret = tensor_argmax(hidden_2, argmax_out);
  printf("arg_ret - %d\n", arg_ret);
  float acc;
  print_shape(d.y);
  print_shape(argmax_out);
  reshape(argmax_out, shapeN(1, 100));
  int acc_ret = accuracy(argmax_out, d.y, &acc);
  printf("acc_ret - %d\n", acc_ret);
  printf("%.3f\n", acc);
  float loss = 0.0;
  reshape(hidden_2, shapeN(2, 100, 10));
  print_shape(hidden_2);
  print_shape(d.y);
  int ce_ret = cross_entropy(hidden_2, d.y, &loss);
  printf("%.05f\n", loss);
  printf("ce_ret - %d\n", ce_ret);
  Tensor *hidden_2_grad = tensor_alloc(hidden_2->shape, DTYPE_FLOAT32); 
  int ceb_ret = cross_entropy_backward(hidden_2, d.y, hidden_2_grad);
  printf("ceb_ret - %d\n", ceb_ret);
  tensor_scale_float(hidden_2_grad, 10000.0);
  print_tensor(hidden_2_grad);
  // fflush(stdout);
  // printf("A rank=%zu dims=%zu,%zu,%zu\n", d.x->shape.rank,
  //      d.x->shape.dims[0], d.x->shape.dims[1], d.x->shape.dims[2]);
  // printf("B rank=%zu dims=%zu,%zu,%zu\n", layer1_weight->shape.rank,
  //        layer1_weight->shape.dims[0], layer1_weight->shape.dims[1],
  //        layer1_weight->shape.dims[2]);
  // printf("C rank=%zu dims=%zu,%zu,%zu\n", hidden_1->shape.rank,
  //        hidden_1->shape.dims[0], hidden_1->shape.dims[1],
  //        hidden_1->shape.dims[2]);

  // print_tensor(hidden_1);

  Tensor *l = tensor_alloc(shapeN(3, 2, 2, 2), DTYPE_FLOAT32);
  Tensor *r = tensor_alloc(shapeN(3, 1, 2, 1), DTYPE_FLOAT32);
  Tensor *b = tensor_alloc(shapeN(3, 2, 2, 1), DTYPE_FLOAT32);
  // tensor_arange_float(l);
  // tensor_arange_float(r);
  float l_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float r_values[] = {1, 2};
  float b_values[] = {1, 2, 3, 4};
  memcpy(l->data, l_values, tensor_byte_count(l));
  memcpy(r->data, r_values, tensor_byte_count(r));
  memcpy(b->data, b_values, tensor_byte_count(b));
  Tensor *out = tensor_alloc(shapeN(3, 2, 2, 1), DTYPE_FLOAT32);
  tensor_fill_float(out, 0.0);
  ret = bmm(out, l, r);
  print_tensor(out);
  ret = tensor_add(out, b);

  printf("tensor add ret %d\n", ret);
  print_tensor(out);

  tensor_free(&t1);
  tensor_free(&t2);
  dataset_free(&d);
  return 0;
}
