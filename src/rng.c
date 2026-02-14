#include "rng.h"

void rng_seed(RNG *r, uint64_t seed) {
  r->state = seed ? seed : 0x9E3779B97F4A7C15ull;
}

uint64_t xorshift64star(uint64_t *s) {
  uint64_t x = *s;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *s = x;
  return x * 2685821657736338717ull;
}

float rng_uniform(RNG *r) {
  uint64_t x = xorshift64star(&r->state);
  uint32_t top24 = (uint32_t)(x >> 40);

  return (float)top24 * (1.0f / 16777216.0f);
}

float rng_normal(RNG *r) {
  float u1;
  float u2 = rng_uniform(r);
  do {
    u1 = rng_uniform(r);
  } while (u1 == 0);

  float mag = sqrtf(-2.0f * logf(u1));
  float ang = 6.2831853071795864769f * u2;

  return mag * cosf(ang);
}
