#pragma once

#include <cstdint>
#include "libcgt/cuda/ThreadMath.cuh"

template< typename T >
struct KernelQueue
{
    __inline__ __host__ __device__
    KernelQueue();

    __inline__ __host__ __device__
    KernelQueue( uint2* d_pReadIndexAndCount, KernelArray1D< T > elements );

    // enqueues a value at the tail the queue
    // (atomically increment the tail pointer
    // and set the value where it used to be)
    // returns the index in ringBuffer() where the element was placed
    // TODO: handling queue full is annoying
    __inline__ __device__
    uint32_t enqueue( const T& val );

    // removes an element from the head of the queue
    // (atomically increment the head pointer
    // and return the value where it used to be)
    // TODO: handling queue empty is annoying
    __inline__ __device__
    T dequeue();

    // enqueues n elements for writing
    // returning a pointer to the first block
    // n > 0
    // (atomically increments the tail pointer by n)
    // TODO: handling queue full is annoying
    __inline__ __device__
    T* enqueueN( uint32_t n = 1 );

    // dequeues n elements for reading
    // n > 0
    // (atomically increments the tail pointer by n)
    // TODO: handling queue empty is annoying
    __inline__ __device__
    T* dequeueN( uint32_t n = 1 );

    __inline__ __device__
    int capacity() const;

    __inline__ __device__
    int count();

    __inline__ __device__
    bool isEmpty();

    __inline__ __device__
    bool isFull();

    __inline__ __device__
    KernelArray1D< T >& ringBuffer();

    // pointer to the head index
    // &( md_pHeadTailAbsoluteIndices->x )
    __inline__ __device__
        uint32_t* headIndexPointer();

    // pointer to the tail index
    // &( md_pHeadTailAbsoluteIndices->y )
    __inline__ __device__
    uint32_t* tailIndexPointer();

private:

    uint2* md_pHeadTailAbsoluteIndices;
    KernelArray1D< T > md_ringBuffer;

};

#include "libcgt/cuda/KernelQueue.inl"
