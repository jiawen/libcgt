#pragma once

template< typename T >
struct KernelVector
{
    T* pointer;
    int length;

    __inline__ __host__ __device__
    KernelVector();

    __inline__ __host__ __device__
    KernelVector( T* _pointer, int _length );

    __inline__ __device__
    const T& operator [] ( int i ) const;

    __inline__ __device__
    T& operator [] ( int i );
};

#include "KernelVector.inl"
