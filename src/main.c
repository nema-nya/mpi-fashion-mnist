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

void test_bmm() {
    Tensor* x = tensor_alloc(shapeN(3, 1, 2, 2), DTYPE_FLOAT32);
    Tensor* y = tensor_alloc(shapeN(3, 1, 2, 2), DTYPE_FLOAT32);
    Tensor* z = tensor_alloc(shapeN(3, 1, 2, 2), DTYPE_FLOAT32);
    float y_values[] = {1, 2, 3, 4};
    float x_values[] = {1, 2, 3, 4};
    float z_values[] = {0, 0, 0, 0};
    memcpy(x->data, x_values, tensor_byte_count(x));
    memcpy(y->data, y_values, tensor_byte_count(y));
    memcpy(z->data, z_values, tensor_byte_count(z));

    int ret = bmm(z, x, y, /*transpose_A=*/false, /*transpose_B=*/false);

    assert(ret == 0);
    float* z_data = (float*)z->data;
    // float z_targets[] = {5, 11, 11, 25}; // B.T
    // float z_targets[] = {10, 14, 14, 20}; // A.T
    // float z_targets[] = {7, 15, 10, 22}; // A.T B.T
    float z_targets[] = {7, 10, 15, 22};
    bool passing = true;
    for (size_t i = 0; i < z->size; ++i) {
        if (fabs(z_data[i] - z_targets[i]) > 1e-5) {
            passing = false;
        }
    }
    if (!passing) {
        printf("Got: ");
        for (size_t i = 0; i < z->size; ++i) {
            printf("%f ", z_data[i]);
        }
        printf("\r\nExpected: ");
        for (size_t i = 0; i < z->size; ++i) {
            printf("%f ", z_targets[i]);
        }
        printf("\r\n");
    }
    assert(passing);
}

void test_bmm_transpose_A() {
    Tensor* x = tensor_alloc(shapeN(3, 1, 2, 2), DTYPE_FLOAT32);
    Tensor* y = tensor_alloc(shapeN(3, 1, 2, 2), DTYPE_FLOAT32);
    Tensor* z = tensor_alloc(shapeN(3, 1, 2, 2), DTYPE_FLOAT32);
    float y_values[] = {1, 2, 3, 4};
    float x_values[] = {1, 2, 3, 4};
    float z_values[] = {0, 0, 0, 0};
    memcpy(x->data, x_values, tensor_byte_count(x));
    memcpy(y->data, y_values, tensor_byte_count(y));
    memcpy(z->data, z_values, tensor_byte_count(z));

    int ret = bmm(z, x, y, /*transpose_A=*/true, /*transpose_B=*/false);

    assert(ret == 0);
    float* z_data = (float*)z->data;
    float z_targets[] = {10, 14, 14, 20};
    bool passing = true;
    for (size_t i = 0; i < z->size; ++i) {
        if (fabs(z_data[i] - z_targets[i]) > 1e-5) {
            passing = false;
        }
    }
    if (!passing) {
        printf("Got: ");
        for (size_t i = 0; i < z->size; ++i) {
            printf("%f ", z_data[i]);
        }
        printf("\r\nExpected: ");
        for (size_t i = 0; i < z->size; ++i) {
            printf("%f ", z_targets[i]);
        }
        printf("\r\n");
    }
    assert(passing);
}

void test_bmm_transpose_B() {
    Tensor* x = tensor_alloc(shapeN(3, 1, 2, 2), DTYPE_FLOAT32);
    Tensor* y = tensor_alloc(shapeN(3, 1, 2, 2), DTYPE_FLOAT32);
    Tensor* z = tensor_alloc(shapeN(3, 1, 2, 2), DTYPE_FLOAT32);
    float y_values[] = {1, 2, 3, 4};
    float x_values[] = {1, 2, 3, 4};
    float z_values[] = {0, 0, 0, 0};
    memcpy(x->data, x_values, tensor_byte_count(x));
    memcpy(y->data, y_values, tensor_byte_count(y));
    memcpy(z->data, z_values, tensor_byte_count(z));

    int ret = bmm(z, x, y, /*transpose_A=*/false, /*transpose_B=*/true);

    assert(ret == 0);
    float* z_data = (float*)z->data;
    float z_targets[] = {5, 11, 11, 25};
    bool passing = true;
    for (size_t i = 0; i < z->size; ++i) {
        if (fabs(z_data[i] - z_targets[i]) > 1e-5) {
            passing = false;
        }
    }
    if (!passing) {
        printf("Got: ");
        for (size_t i = 0; i < z->size; ++i) {
            printf("%f ", z_data[i]);
        }
        printf("\r\nExpected: ");
        for (size_t i = 0; i < z->size; ++i) {
            printf("%f ", z_targets[i]);
        }
        printf("\r\n");
    }
    assert(passing);
}

void test_bmm_transpose_AB() {
    Tensor* x = tensor_alloc(shapeN(3, 1, 2, 2), DTYPE_FLOAT32);
    Tensor* y = tensor_alloc(shapeN(3, 1, 2, 2), DTYPE_FLOAT32);
    Tensor* z = tensor_alloc(shapeN(3, 1, 2, 2), DTYPE_FLOAT32);
    float y_values[] = {1, 2, 3, 4};
    float x_values[] = {1, 2, 3, 4};
    float z_values[] = {0, 0, 0, 0};
    memcpy(x->data, x_values, tensor_byte_count(x));
    memcpy(y->data, y_values, tensor_byte_count(y));
    memcpy(z->data, z_values, tensor_byte_count(z));

    int ret = bmm(z, x, y, /*transpose_A=*/true, /*transpose_B=*/true);

    assert(ret == 0);
    float* z_data = (float*)z->data;
    float z_targets[] = {7, 15, 10, 22};
    bool passing = true;
    for (size_t i = 0; i < z->size; ++i) {
        if (fabs(z_data[i] - z_targets[i]) > 1e-5) {
            passing = false;
        }
    }
    if (!passing) {
        printf("Got: ");
        for (size_t i = 0; i < z->size; ++i) {
            printf("%f ", z_data[i]);
        }
        printf("\r\nExpected: ");
        for (size_t i = 0; i < z->size; ++i) {
            printf("%f ", z_targets[i]);
        }
        printf("\r\n");
    }
    assert(passing);
}

void test_bmm_transpose_A_fuzz() {
    Tensor* x = tensor_alloc(shapeN(3, 1, 64, 64), DTYPE_FLOAT32);
    Tensor* x_perm = tensor_alloc(shapeN(3, 1, 64, 64), DTYPE_FLOAT32);
    Tensor* y = tensor_alloc(shapeN(3, 1, 64, 64), DTYPE_FLOAT32);
    Tensor* z1 = tensor_alloc(shapeN(3, 1, 64, 64), DTYPE_FLOAT32);
    Tensor* z2 = tensor_alloc(shapeN(3, 1, 64, 64), DTYPE_FLOAT32);
    RNG rng;
    rng.state = 42;
    tensor_fill_rand_normal(x, &rng);
    tensor_fill_rand_normal(y, &rng);

    int ret = 0;
    ret = bmm(z1, x, y, /*transpose_A=*/true, /*transpose_B=*/false);
    assert(ret == 0);

    tensor_copy(x_perm, x);
    permute(x_perm, 0, 2, 1);
    ret = bmm(z2, x_perm, y, /*transpose_A=*/false, /*transpose_B=*/false);
    assert(ret == 0);

    float* z1_data = (float*)z1->data;
    float* z2_data = (float*)z2->data;
    for (size_t i = 0; i < z1->size; ++i) {
        assert(fabs(z1_data[i] - z2_data[i]) < 1e-5);
    }
}

void test_bmm_transpose_B_fuzz() {
    Tensor* x = tensor_alloc(shapeN(3, 1, 64, 64), DTYPE_FLOAT32);
    Tensor* y = tensor_alloc(shapeN(3, 1, 64, 64), DTYPE_FLOAT32);
    Tensor* y_perm = tensor_alloc(shapeN(3, 1, 64, 64), DTYPE_FLOAT32);
    Tensor* z1 = tensor_alloc(shapeN(3, 1, 64, 64), DTYPE_FLOAT32);
    Tensor* z2 = tensor_alloc(shapeN(3, 1, 64, 64), DTYPE_FLOAT32);
    RNG rng;
    rng.state = 42;
    tensor_fill_rand_normal(x, &rng);
    tensor_fill_rand_normal(y, &rng);

    int ret = 0;
    ret = bmm(z1, x, y, /*transpose_A=*/false, /*transpose_B=*/true);
    assert(ret == 0);

    tensor_copy(y_perm, y);
    permute(y_perm, 0, 2, 1);
    ret = bmm(z2, x, y_perm, /*transpose_A=*/false, /*transpose_B=*/false);
    assert(ret == 0);

    float* z1_data = (float*)z1->data;
    float* z2_data = (float*)z2->data;
    for (size_t i = 0; i < z1->size; ++i) {
        assert(fabs(z1_data[i] - z2_data[i]) < 1e-5);
    }
}

bool verify_endianness() {
    uint16_t dummy = 0x0100;
    uint8_t* dummy_ptr = (uint8_t*)(&dummy);
    return dummy_ptr[0] == 0 && dummy_ptr[1] == 1;
}

int main(void) {
    test_bcast();
    test_bmm();
    test_bmm_transpose_A();
    test_bmm_transpose_B();
    test_bmm_transpose_AB();
    test_bmm_transpose_A_fuzz();
    test_bmm_transpose_B_fuzz();
    assert(("Your system is big-endian", verify_endianness()));
    Dataset d;
    if (dataset_load_bin("data/train-labels.bin", "data/train-data.bin", &d)) {
        printf("failed to load dataset\n");
        return 1;
    }
    Tensor* new_x = tensor_alloc(shapeN(3, 100, 1, 784), DTYPE_FLOAT32);
    Tensor* new_y = tensor_alloc(shapeN(1, 100), DTYPE_UINT8);

    reshape(d.x, shapeN(3, 60000, 1, 784));
    RETURN_IF_ERROR(tensor_slice(d.x, new_x, 0, 0, 100));
    RETURN_IF_ERROR(tensor_slice(d.y, new_y, 0, 0, 100));
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

    Tensor* layer1_weight = tensor_alloc(shapeN(2, 784, 256), DTYPE_FLOAT32);
    Tensor* layer1_bias = tensor_alloc(shapeN(1, 256), DTYPE_FLOAT32);
    Tensor* layer2_weight = tensor_alloc(shapeN(2, 256, 10), DTYPE_FLOAT32);
    Tensor* layer2_bias = tensor_alloc(shapeN(1, 10), DTYPE_FLOAT32);

    RETURN_IF_ERROR(reshape(layer1_weight, shapeN(3, 1, 784, 256)));
    RETURN_IF_ERROR(reshape(d.x, shapeN(3, 100, 1, 784)));
    RETURN_IF_ERROR(reshape(layer2_weight, shapeN(3, 1, 256, 10)));

    RNG r;
    r.state = 67;

    tensor_fill_uniform(layer1_weight, &r);
    tensor_fill_uniform(layer1_bias, &r);
    tensor_fill_uniform(layer2_weight, &r);
    tensor_fill_uniform(layer2_bias, &r);

    float k1 = sqrtf(1.0 / 784.0);
    float k2 = sqrtf(1.0 / 256.0);

    tensor_scale_and_add_const(layer1_weight, 2 * k1, -k1);
    tensor_scale_and_add_const(layer1_bias, 2 * k1, -k1);

    tensor_scale_and_add_const(layer2_weight, 2 * k2, -k2);
    tensor_scale_and_add_const(layer2_bias, 2 * k2, -k2);

    // forward
    Tensor* hidden_1 = tensor_alloc(shapeN(3, 100, 1, 256), DTYPE_FLOAT32);
    Tensor* hidden_1_grad = tensor_alloc(shapeN(3, 100, 1, 256), DTYPE_FLOAT32);
    Tensor* hidden_1_pre_tanh = tensor_alloc(hidden_1->shape, DTYPE_FLOAT32);
    Tensor* layer1_bias_grad = tensor_alloc(layer1_bias->shape, DTYPE_FLOAT32);
    Tensor* layer1_weight_grad =
        tensor_alloc(layer1_weight->shape, DTYPE_FLOAT32);
    Tensor* layer1_weight_m = tensor_alloc(layer1_weight->shape, DTYPE_FLOAT32);
    Tensor* layer1_weight_v = tensor_alloc(layer1_weight->shape, DTYPE_FLOAT32);
    Tensor* layer1_bias_m = tensor_alloc(layer1_bias->shape, DTYPE_FLOAT32);
    Tensor* layer1_bias_v = tensor_alloc(layer1_bias->shape, DTYPE_FLOAT32);
    tensor_fill_float(layer1_weight_m, 0.0f);
    tensor_fill_float(layer1_weight_v, 0.0f);
    tensor_fill_float(layer1_bias_m, 0.0f);
    tensor_fill_float(layer1_bias_v, 0.0f);

    Tensor* argmax_out = tensor_alloc(shapeN(2, 100, 1), DTYPE_UINT8);

    Tensor* hidden_2 = tensor_alloc(shapeN(3, 100, 1, 10), DTYPE_FLOAT32);
    Tensor* hidden_2_grad = tensor_alloc(shapeN(2, 100, 10), DTYPE_FLOAT32);

    Tensor* layer2_bias_grad = tensor_alloc(layer2_bias->shape, DTYPE_FLOAT32);
    Tensor* layer2_weight_grad =
        tensor_alloc(layer2_weight->shape, DTYPE_FLOAT32);
    Tensor* layer2_weight_m = tensor_alloc(layer2_weight->shape, DTYPE_FLOAT32);
    Tensor* layer2_weight_v = tensor_alloc(layer2_weight->shape, DTYPE_FLOAT32);
    Tensor* layer2_bias_m = tensor_alloc(layer2_bias->shape, DTYPE_FLOAT32);
    Tensor* layer2_bias_v = tensor_alloc(layer2_bias->shape, DTYPE_FLOAT32);
    tensor_fill_float(layer2_weight_m, 0.0f);
    tensor_fill_float(layer2_weight_v, 0.0f);
    tensor_fill_float(layer2_bias_m, 0.0f);
    tensor_fill_float(layer2_bias_v, 0.0f);
    float acc = 0.0f;
    float loss = 0.0f;
    size_t t = 0;
    size_t epochs = 1000;
    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-08f;
    for (size_t ep = 0; ep < epochs; ++ep) {
        tensor_fill_float(layer1_weight_grad, 0.0f);
        tensor_fill_float(layer1_bias_grad, 0.0f);
        tensor_fill_float(layer2_weight_grad, 0.0f);
        tensor_fill_float(layer2_bias_grad, 0.0f);
        tensor_fill_float(hidden_1, 0.0f);
        tensor_fill_float(hidden_2, 0.0f);
        tensor_fill_float(hidden_1_grad, 0.0f);

        t++;

        RETURN_IF_ERROR(bmm(hidden_1, d.x, layer1_weight, false, false));
        RETURN_IF_ERROR(tensor_add(hidden_1, layer1_bias));

        RETURN_IF_ERROR(tensor_copy(hidden_1_pre_tanh, hidden_1));

        RETURN_IF_ERROR(tensor_tanh(hidden_1));

        RETURN_IF_ERROR(bmm(hidden_2, hidden_1, layer2_weight, false, false));
        RETURN_IF_ERROR(tensor_add(hidden_2, layer2_bias));

        RETURN_IF_ERROR(tensor_argmax(hidden_2, argmax_out));
        RETURN_IF_ERROR(reshape(argmax_out, shapeN(1, 100)));
        RETURN_IF_ERROR(accuracy(argmax_out, d.y, &acc));
        RETURN_IF_ERROR(reshape(argmax_out, shapeN(2, 100, 1)));

        RETURN_IF_ERROR(reshape(hidden_2, shapeN(2, 100, 10)));
        RETURN_IF_ERROR(cross_entropy(hidden_2, d.y, &loss));
        printf("epoch %zu loss = %.5f acc = %.5f\n", ep, loss, acc);
        ////////////////////////////////////////////////////////////////////////////////////////////
        RETURN_IF_ERROR(cross_entropy_backward(hidden_2, d.y, hidden_2_grad));
        RETURN_IF_ERROR(reshape(hidden_2, shapeN(3, 100, 1, 10)));
        RETURN_IF_ERROR(reshape(hidden_2_grad, shapeN(3, 100, 1, 10)));

        RETURN_IF_ERROR(
            tensor_add_backward(hidden_2_grad, NULL, layer2_bias_grad));
        RETURN_IF_ERROR(bmm_backward(hidden_1, layer2_weight, hidden_2_grad,
                                     hidden_1_grad, layer2_weight_grad));
        RETURN_IF_ERROR(reshape(hidden_2_grad, shapeN(2, 100, 10)));

        RETURN_IF_ERROR(tensor_tanh_backward(hidden_1_pre_tanh, hidden_1_grad));

        RETURN_IF_ERROR(
            tensor_add_backward(hidden_1_grad, NULL, layer1_bias_grad));
        RETURN_IF_ERROR(bmm_backward(d.x, layer1_weight, hidden_1_grad, NULL,
                                     layer1_weight_grad));

        RETURN_IF_ERROR(adam_step(lr, beta1, beta2, eps, t, layer1_weight_grad,
                                  layer1_weight, layer1_weight_m,
                                  layer1_weight_v));

        RETURN_IF_ERROR(adam_step(lr, beta1, beta2, eps, t, layer1_bias_grad,
                                  layer1_bias, layer1_bias_m, layer1_bias_v));

        RETURN_IF_ERROR(adam_step(lr, beta1, beta2, eps, t, layer2_weight_grad,
                                  layer2_weight, layer2_weight_m,
                                  layer2_weight_v));

        RETURN_IF_ERROR(adam_step(lr, beta1, beta2, eps, t, layer2_bias_grad,
                                  layer2_bias, layer2_bias_m, layer2_bias_v));
    }

    tensor_free(&t1);
    tensor_free(&t2);
    dataset_free(&d);

    return 0;
}
