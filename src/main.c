#include <stdio.h>

#include "dataset.h"
#include "linalg.h"
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
    Tensor t3 = {0};
    Shape s = shape3(3, 3, 3);
    // Shape s_t2 = shape3()
    // Shape
    if (!tensor_alloc(&t1, s) || !tensor_alloc(&t2, s) ||
        !tensor_alloc(&t3, s)) {
        printf("failed to allocate tensors\n");
        tensor_free(&t1);
        tensor_free(&t2);
        dataset_free(&d);
        return 1;
    }

    if (!tensor_fill(&t1, 1.5f) || !tensor_fill(&t2, 1.5f) ||
        !tensor_zero(&t3)) {
        printf("tensor operation failed\n");
        tensor_free(&t1);
        tensor_free(&t2);
        dataset_free(&d);
        return 1;
    }
    if (!bmm(&t3, &t1, &t2)) {
        printf("failed\n");
    }
    // RNG rng = {0};
    // rng.state = 67;
    // tensor_fill_rand_normal(&t1, &rng);

    for (size_t i = 0; i < t3.size; ++i) {
        printf("%f\n", t3.data[i]);
    }

    tensor_free(&t1);
    tensor_free(&t2);

    dataset_free(&d);
    return 0;
}
