#pragma once

template< typename T >
struct KernelArray1D
{
    T* pointer;
    int length;

    __inline__ __host__ __device__
    KernelArray1D();

    __inline__ __host__ __device__
    KernelArray1D( T* _pointer, int _length );

    __inline__ __device__
    const T& operator [] ( int i ) const;

    __inline__ __device__
    T& operator [] ( int i );
};

#include "KernelArray1D.inl"
