#include <stdio.h>

#include "dataset.h"

int main(void) {
    Dataset d;

    if (!dataset_load_csv("data/fashion-mnist_train.csv", &d)) {
        printf("failed to load dataset\n");
        return 1;
    }

    printf("loaded %zu samples\n", d.n);

    printf("first label %u\n", (unsigned)d.y[0]);
    printf("first 32 pixels of the first img\n");

    for (int i = 0; i < 32; ++i) {
        printf("%.3f ", d.x[i]);
    }
    printf("\n");

    for (size_t i = 0; i < d.n; ++i) {
        printf("%zu label %u\n", i, (unsigned)d.y[i]);
    }
    dataset_free(&d);
    return 0;
}
