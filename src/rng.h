#ifndef RNG_H
#define RNG_H

#include <math.h>
#include <stdint.h>

typedef struct {
    uint64_t state;
} RNG;

void rng_seed(RNG* r, uint64_t seed);

uint64_t xorshift64star(uint64_t* s);

float rng_uniform(RNG* r);

float rng_normal(RNG* r);

#endif