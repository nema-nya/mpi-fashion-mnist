#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"
#include "linalg.h"
#include "optim.h"
#include "tensor.h"
#include "utils.h"

void test_bcast() {
    Tensor* y = tensor_alloc(shapeN(3, 2, 2, 2), DTYPE_FLOAT32);
    Tensor* x = tensor_alloc(shapeN(2, 1, 2), DTYPE_FLOAT32);
    float y_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float x_values[] = {1, 2};
    memcpy(y->data, y_values, tensor_byte_count(y));
    memcpy(x->data, x_values, tensor_byte_count(x));

    int ret = tensor_bcast_grad(y, x);

    assert(ret == 0);
    float* x_data = (float*)x->data;
    assert(fabs(x_data[0] - 16.0) < 1e-5);
    assert(fabs(x_data[1] - 20.0) < 1e-5);
}

bool verify_endianness() {
    uint16_t dummy = 0x0100;
    uint8_t* dummy_ptr = (uint8_t*)(&dummy);
    return dummy_ptr[0] == 0 && dummy_ptr[1] == 1;
}

int main(void) {
    test_bcast();
    assert(("Your system is big-endian", verify_endianness()));
    Dataset d;
    if (dataset_load_bin("data/train-labels.bin", "data/train-data.bin", &d)) {
        printf("failed to load dataset\n");
        return 1;
    }
    Tensor* new_x = tensor_alloc(shapeN(3, 100, 1, 784), DTYPE_FLOAT32);
    Tensor* new_y = tensor_alloc(shapeN(1, 100), DTYPE_UINT8);

    reshape(d.x, shapeN(3, 60000, 1, 784));
    int ts_ret1 = tensor_slice(d.x, new_x, 0, 0, 100);
    int ts_ret2 = tensor_slice(d.y, new_y, 0, 0, 100);
    free(d.x);
    free(d.y);
    d.x = new_x;
    d.y = new_y;

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

    float* layer1_weight_data =
        (float*)read_all("data/layer1-weight.bin", NULL);
    float* layer1_bias_data = (float*)read_all("data/layer1-bias.bin", NULL);
    float* layer2_weight_data =
        (float*)read_all("data/layer2-weight.bin", NULL);
    float* layer2_bias_data = (float*)read_all("data/layer2-bias.bin", NULL);

    Tensor* layer1_weight = tensor_alloc(shapeN(2, 784, 256), DTYPE_FLOAT32);
    Tensor* layer1_bias = tensor_alloc(shapeN(1, 256), DTYPE_FLOAT32);
    Tensor* layer2_weight = tensor_alloc(shapeN(2, 256, 10), DTYPE_FLOAT32);
    Tensor* layer2_bias = tensor_alloc(shapeN(1, 10), DTYPE_FLOAT32);

    free(layer1_weight->data);
    free(layer1_bias->data);
    free(layer2_weight->data);
    free(layer2_bias->data);

    layer1_weight->data = (void*)layer1_weight_data;
    layer1_bias->data = (void*)layer1_bias_data;
    layer2_weight->data = (void*)layer2_weight_data;
    layer2_bias->data = (void*)layer2_bias_data;

    RETURN_IF_ERROR(reshape(layer1_weight, shapeN(3, 1, 784, 256)));
    RETURN_IF_ERROR(reshape(d.x, shapeN(3, 100, 1, 784)));
    RETURN_IF_ERROR(reshape(layer2_weight, shapeN(3, 1, 256, 10)));

    // forward

    Tensor* hidden_1 = tensor_alloc(shapeN(3, 100, 1, 256), DTYPE_FLOAT32);
    Tensor* hidden_1_grad = tensor_alloc(shapeN(3, 100, 1, 256), DTYPE_FLOAT32);
    Tensor* hidden_1_pre_tanh = tensor_alloc(hidden_1->shape, DTYPE_FLOAT32);
    Tensor* layer1_bias_grad = tensor_alloc(layer1_bias->shape, DTYPE_FLOAT32);
    Tensor* layer1_weight_grad =
        tensor_alloc(layer1_weight->shape, DTYPE_FLOAT32);

    Tensor* argmax_out = tensor_alloc(shapeN(2, 100, 1), DTYPE_UINT8);

    Tensor* hidden_2 = tensor_alloc(shapeN(3, 100, 1, 10), DTYPE_FLOAT32);
    Tensor* hidden_2_grad = tensor_alloc(shapeN(2, 100, 10), DTYPE_FLOAT32);

    Tensor* layer2_bias_grad = tensor_alloc(layer2_bias->shape, DTYPE_FLOAT32);
    Tensor* layer2_weight_grad =
        tensor_alloc(shapeN(3, 100, 256, 10), DTYPE_FLOAT32);
    float acc = 0.0;
    float loss = 0.0;

    RETURN_IF_ERROR(bmm(hidden_1, d.x, layer1_weight, false, false));
    RETURN_IF_ERROR(tensor_add(hidden_1, layer1_bias));

    RETURN_IF_ERROR(tensor_copy(hidden_1_pre_tanh, hidden_1));

    RETURN_IF_ERROR(tensor_tanh(hidden_1));

    RETURN_IF_ERROR(bmm(hidden_2, hidden_1, layer2_weight, false, false));
    RETURN_IF_ERROR(tensor_add(hidden_2, layer2_bias));

    RETURN_IF_ERROR(tensor_argmax(hidden_2, argmax_out));
    RETURN_IF_ERROR(reshape(argmax_out, shapeN(1, 100)));
    RETURN_IF_ERROR(accuracy(argmax_out, d.y, &acc));

    RETURN_IF_ERROR(reshape(hidden_2, shapeN(2, 100, 10)));
    RETURN_IF_ERROR(cross_entropy(hidden_2, d.y, &loss));
    printf("%.5f\n", loss);
    ////////////////////////////////////////////////////////////////////////////////////////////
    RETURN_IF_ERROR(cross_entropy_backward(hidden_2, d.y, hidden_2_grad));
    RETURN_IF_ERROR(reshape(hidden_2, shapeN(3, 100, 1, 10)));
    RETURN_IF_ERROR(reshape(hidden_2_grad, shapeN(3, 100, 1, 10)));

    RETURN_IF_ERROR(tensor_add_backward(hidden_2_grad, NULL, layer2_bias_grad));
    RETURN_IF_ERROR(bmm_backward(hidden_1, layer2_weight, hidden_2_grad,
                                 hidden_1_grad, layer2_weight_grad));

    RETURN_IF_ERROR(tensor_tanh_backward(hidden_1_pre_tanh, hidden_1_grad));

    RETURN_IF_ERROR(tensor_add_backward(hidden_1_grad, NULL, layer1_bias_grad));
    RETURN_IF_ERROR(bmm_backward(d.x, layer1_weight, hidden_1_grad, NULL,
                                 layer1_weight_grad));

    print_tensor(layer1_weight_grad);
    print_tensor(layer1_bias_grad);
    RETURN_IF_ERROR(tensor_scale_float(layer2_weight_grad, 10000.0f));
    print_tensor(layer2_weight_grad);
    print_tensor(layer2_bias_grad);

    tensor_free(&t1);
    tensor_free(&t2);
    dataset_free(&d);

    return 0;
}
