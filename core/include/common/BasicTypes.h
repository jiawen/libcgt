#pragma once

#include <cstdint>
#include <cstdio>

#ifndef int8x2
typedef struct
{
    int8_t x;
    int8_t y;
} int8x2;
#endif

#ifndef int8x3
typedef struct
{
    int8_t x;
    int8_t y;
    int8_t z;
} int8x3;
#endif

#ifndef int8x4
typedef struct
{
    int8_t x;
    int8_t y;
    int8_t z;
    int8_t w;
} int8x4;
#endif

#ifndef uint8x2
typedef struct
{
	uint8_t x;
	uint8_t y;
} uint8x2;
#endif

#ifndef uint8x3
typedef struct
{
	uint8_t x;
	uint8_t y;
	uint8_t z;
} uint8x3;
#endif

#ifndef uint8x4
typedef struct
{
	uint8_t x;
	uint8_t y;
	uint8_t z;
	uint8_t w;
} uint8x4;
#endif
