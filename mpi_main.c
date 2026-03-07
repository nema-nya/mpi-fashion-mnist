#include <assert.h>
#include <math.h>
#include <mpi.h>
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

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    Dataset d;
    RETURN_IF_ERROR(
        dataset_load_bin("data/train-labels.bin", "data/train-data.bin", &d));
    float frac_to_take = 0.01;
    int n_of_data_to_take = 60000 * frac_to_take;

    Tensor* new_x =
        tensor_alloc(shapeN(3, n_of_data_to_take, 1, 784), DTYPE_FLOAT32);
    Tensor* new_y = tensor_alloc(shapeN(1, n_of_data_to_take), DTYPE_UINT8);

    reshape(d.x, shapeN(3, 60000, 1, 784));
    RETURN_IF_ERROR(tensor_slice(d.x, new_x, 0, 0, n_of_data_to_take));
    RETURN_IF_ERROR(tensor_slice(d.y, new_y, 0, 0, n_of_data_to_take));
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

    float acc = 0.0f;
    float loss = 0.0f;
    size_t t = 0;
    size_t epochs = 5;
    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-08f;
    int batch_size = 8;

    Tensor* layer1_weight = tensor_alloc(shapeN(2, 784, 256), DTYPE_FLOAT32);
    Tensor* layer1_bias = tensor_alloc(shapeN(1, 256), DTYPE_FLOAT32);
    Tensor* layer2_weight = tensor_alloc(shapeN(2, 256, 10), DTYPE_FLOAT32);
    Tensor* layer2_bias = tensor_alloc(shapeN(1, 10), DTYPE_FLOAT32);

    RETURN_IF_ERROR(reshape(layer1_weight, shapeN(3, 1, 784, 256)));
    RETURN_IF_ERROR(reshape(d.x, shapeN(3, n_of_data_to_take, 1, 784)));
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
    Tensor* hidden_1 =
        tensor_alloc(shapeN(3, batch_size, 1, 256), DTYPE_FLOAT32);
    Tensor* hidden_1_grad =
        tensor_alloc(shapeN(3, batch_size, 1, 256), DTYPE_FLOAT32);
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

    Tensor* argmax_out = tensor_alloc(shapeN(2, batch_size, 1), DTYPE_UINT8);

    Tensor* hidden_2 =
        tensor_alloc(shapeN(3, batch_size, 1, 10), DTYPE_FLOAT32);
    Tensor* hidden_2_grad =
        tensor_alloc(shapeN(2, batch_size, 10), DTYPE_FLOAT32);

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
    Shape d_x_shape = d.x->shape;
    d_x_shape.dims[0] = batch_size;
    Shape d_y_shape = d.y->shape;
    d_y_shape.dims[0] = batch_size;
    Tensor* batch_x = tensor_alloc(d_x_shape, DTYPE_FLOAT32);
    Tensor* batch_y = tensor_alloc(d_y_shape, DTYPE_UINT8);
    for (size_t ep = 0; ep < epochs; ++ep) {
        dataset_rand_perm(d.x, d.y, &r);
        for (size_t batch = 0;
             batch < n_of_data_to_take - batch_size * world_size;
             batch += batch_size * world_size) {
            RETURN_IF_ERROR(
                tensor_slice(d.x, batch_x, 0, batch + world_rank * batch_size,
                             batch + world_rank * batch_size + batch_size));
            RETURN_IF_ERROR(
                tensor_slice(d.y, batch_y, 0, batch + world_rank * batch_size,
                             batch + world_rank * batch_size + batch_size));
            tensor_fill_float(layer1_weight_grad, 0.0f);
            tensor_fill_float(layer1_bias_grad, 0.0f);
            tensor_fill_float(layer2_weight_grad, 0.0f);
            tensor_fill_float(layer2_bias_grad, 0.0f);
            tensor_fill_float(hidden_1, 0.0f);
            tensor_fill_float(hidden_2, 0.0f);
            tensor_fill_float(hidden_1_grad, 0.0f);

            t++;

            RETURN_IF_ERROR(
                bmm(hidden_1, batch_x, layer1_weight, false, false));
            RETURN_IF_ERROR(tensor_add(hidden_1, layer1_bias));

            RETURN_IF_ERROR(tensor_copy(hidden_1_pre_tanh, hidden_1));

            RETURN_IF_ERROR(tensor_tanh(hidden_1));

            RETURN_IF_ERROR(
                bmm(hidden_2, hidden_1, layer2_weight, false, false));
            RETURN_IF_ERROR(tensor_add(hidden_2, layer2_bias));

            RETURN_IF_ERROR(tensor_argmax(hidden_2, argmax_out));
            RETURN_IF_ERROR(reshape(argmax_out, shapeN(1, batch_size)));
            RETURN_IF_ERROR(accuracy(argmax_out, batch_y, &acc));
            RETURN_IF_ERROR(reshape(argmax_out, shapeN(2, batch_size, 1)));

            RETURN_IF_ERROR(reshape(hidden_2, shapeN(2, batch_size, 10)));
            RETURN_IF_ERROR(cross_entropy(hidden_2, batch_y, &loss));
            printf("%d epoch %zu loss = %.5f acc = %.5f\n", world_rank, ep,
                   loss, acc);
            ////////////////////////////////////////////////////////////////////////////////////////////
            RETURN_IF_ERROR(
                cross_entropy_backward(hidden_2, batch_y, hidden_2_grad));
            RETURN_IF_ERROR(reshape(hidden_2, shapeN(3, batch_size, 1, 10)));
            RETURN_IF_ERROR(
                reshape(hidden_2_grad, shapeN(3, batch_size, 1, 10)));

            RETURN_IF_ERROR(
                tensor_add_backward(hidden_2_grad, NULL, layer2_bias_grad));
            RETURN_IF_ERROR(bmm_backward(hidden_1, layer2_weight, hidden_2_grad,
                                         hidden_1_grad, layer2_weight_grad));
            RETURN_IF_ERROR(reshape(hidden_2_grad, shapeN(2, batch_size, 10)));

            RETURN_IF_ERROR(
                tensor_tanh_backward(hidden_1_pre_tanh, hidden_1_grad));

            RETURN_IF_ERROR(
                tensor_add_backward(hidden_1_grad, NULL, layer1_bias_grad));
            RETURN_IF_ERROR(bmm_backward(batch_x, layer1_weight, hidden_1_grad,
                                         NULL, layer1_weight_grad));

            MPI_Allreduce(MPI_IN_PLACE, layer1_weight_grad->data,
                          layer1_weight_grad->size, MPI_FLOAT, MPI_SUM,
                          MPI_COMM_WORLD);

            MPI_Allreduce(MPI_IN_PLACE, layer1_bias_grad->data,
                          layer1_bias_grad->size, MPI_FLOAT, MPI_SUM,
                          MPI_COMM_WORLD);

            MPI_Allreduce(MPI_IN_PLACE, layer2_weight_grad->data,
                          layer2_weight_grad->size, MPI_FLOAT, MPI_SUM,
                          MPI_COMM_WORLD);

            MPI_Allreduce(MPI_IN_PLACE, layer2_bias_grad->data,
                          layer2_bias_grad->size, MPI_FLOAT, MPI_SUM,
                          MPI_COMM_WORLD);

            RETURN_IF_ERROR(adam_step(lr, beta1, beta2, eps, t,
                                      layer1_weight_grad, layer1_weight,
                                      layer1_weight_m, layer1_weight_v));

            RETURN_IF_ERROR(adam_step(lr, beta1, beta2, eps, t,
                                      layer1_bias_grad, layer1_bias,
                                      layer1_bias_m, layer1_bias_v));

            RETURN_IF_ERROR(adam_step(lr, beta1, beta2, eps, t,
                                      layer2_weight_grad, layer2_weight,
                                      layer2_weight_m, layer2_weight_v));

            RETURN_IF_ERROR(adam_step(lr, beta1, beta2, eps, t,
                                      layer2_bias_grad, layer2_bias,
                                      layer2_bias_m, layer2_bias_v));
        }
    }
    MPI_Finalize();
    return 0;
}
