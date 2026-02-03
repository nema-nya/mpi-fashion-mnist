#include <stdio.h>

#include "dataset.h"
#include "tensor.h"

int main(void) {
    Dataset d;

    if (!dataset_load_csv("data/fashion-mnist_train.csv", &d)) {
        printf("failed to load dataset\n");
        return 1;
    }

    printf("loaded %zu samples\n", d.n);

    Tensor t1 = {0};
    Tensor t2 = {0};
    Shape s = shape2(3, 5);
    if (!tensor_alloc(&t1, s) || !tensor_alloc(&t2, s)) {
        printf("failed to allocate tensors\n");
        tensor_free(&t1);
        tensor_free(&t2);
        dataset_free(&d);
        return 1;
    }

    if (!tensor_zero(&t1) || !tensor_fill(&t2, 1.123f) ||
        !tensor_add(&t1, &t2) || !tensor_mul(&t1, &t2)) {
        printf("tensor operation failed\n");
        tensor_free(&t1);
        tensor_free(&t2);
        dataset_free(&d);
        return 1;
    }
    RNG rng = {0};
    rng.state = 67;
    tensor_fill_rand_normal(&t1, &rng);

    for (size_t i = 0; i < t1.size; ++i) {
        printf("%f\n", t1.data[i]);
    }

    tensor_free(&t1);
    tensor_free(&t2);

    dataset_free(&d);
    return 0;
}
