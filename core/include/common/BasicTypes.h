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

#ifndef int16x2
typedef struct
{
    int16_t x;
    int16_t y;
} int16x2;
#endif

#ifndef int16x3
typedef struct
{
    int16_t x;
    int16_t y;
    int16_t z;
} int16x3;
#endif

#ifndef int16x4
typedef struct
{
    int16_t x;
    int16_t y;
    int16_t z;
    int16_t w;
} int16x4;
#endif

#ifndef uint16x2
typedef struct
{
    uint16_t x;
    uint16_t y;
} uint16x2;
#endif

#ifndef uint16x3
typedef struct
{
    uint16_t x;
    uint16_t y;
    uint16_t z;
} uint16x3;
#endif

#ifndef uint16x4
typedef struct
{
    uint16_t x;
    uint16_t y;
    uint16_t z;
    uint16_t w;
} uint16x4;
#endif