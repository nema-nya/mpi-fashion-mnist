#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>

void return_if_error_body(int code, char* msg);

#define RETURN_IF_ERROR(expr)                       \
    {                                               \
        int ___retval = (expr);                     \
        if (___retval != 0) {                       \
            return_if_error_body(___retval, #expr); \
            return ___retval;                       \
        }                                           \
    }

#define CHECK(expr)                        \
    {                                      \
        if (!(expr)) {                     \
            printf("%s = false\n", #expr); \
            return 1;                      \
        }                                  \
    }

void* read_all(const char* path, size_t* n);

#endif
