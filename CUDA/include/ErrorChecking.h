#pragma once

#include <cuda.h>

#if CUDART_VERSION >= 4000
#define CUT_DEVICE_SYNCHRONIZE( )   cudaDeviceSynchronize();
#else
#define CUT_DEVICE_SYNCHRONIZE( )   cudaThreadSynchronize();
#endif

#ifdef _DEBUG
#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
    errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
    exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    err = CUT_DEVICE_SYNCHRONIZE();                                           \
    if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
    errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
    exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }
#else
#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
    errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
    exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }
#endif

