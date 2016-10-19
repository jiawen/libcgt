#pragma once

#include <cassert>

#include "DeviceVariable.h"
#include "DeviceArray1D.h"

#include "MathUtils.h"
#include "KernelQueue.h"

// TODO: reserveEnqueue( n, bool predicate ), commitEnqueue( n, bool predicate )
// returns a pointer for each thread, predicate has to eavluate to be true for...
// same for dequeue: to relieve pressure on the atomic

// An atomic producer-consumer queue for CUDA
// implemented using a circular buffer
template< typename T >
class DeviceQueue
{
public:

    // nitializes a null queue
    DeviceQueue();

    // nitializes a queue with a fixed capacity
    DeviceQueue( size_t capacity );

    DeviceQueue( const DeviceQueue< T >& copy );
    DeviceQueue( DeviceQueue< T >&& move );
    DeviceQueue< T >& operator = ( const DeviceQueue< T >& copy );
    DeviceQueue< T >& operator = ( DeviceQueue< T >&& move );
    // default destructor

    bool isNull() const;
    bool notNull() const;

    // Returns the maximum number of elements the queue can hold.
    size_t capacity() const;

    // returns the total size of the queue in bytes, including overhead
    size_t sizeInBytes() const;

    // resizes the queue:
    //   destroys the existing data
    //   and clears the queue in the process (head and tail pointers are set to 0)
    void resize( size_t capacity );

    // clears the queue by setting head and tail absolute indices back to 0
    // does not actually touch the data
    void clear( int headIndex = 0, int tailIndex = 0 );

    bool isEmpty() const;
    bool isFull() const;

    // number of enqueued items
    // WARNING: slightly expensive query: incurs a copy from GPU
    int count() const;

    // returns the absolute indices of the head and tail of the queue
    // mod by capacity() to get indices within ringBuffer()
    // WARNING: slightly expensive: incurs a copy from GPU
    uint2 headAndTailAbsoluteIndices() const;

    void setHeadAbsoluteIndex( int headIndex );
    void setTailAbsoluteIndex( int tailIndex );
    void setHeadAndTailAbsoluteIndices( int headIndex, int tailIndex );
    void setHeadAndTailAbsoluteIndices( const uint2& ht );

    DeviceArray1D< T >& ringBuffer();

    KernelQueue< T > kernelQueue();

    // enqueues a value at the tail the queue from the host
    // returns false if it fails (the queue is full)
    // WARNING: can be expensive: incurs copies to and from the GPU
    bool enqueueFromHost( const T& val );

    // dequeues a value at the head of the queue
    // and copies it to the host
    // returns false if it fails (the queue is empty)
    // WARNING: can be expensive: incurs copies to and from the GPU
    bool dequeueToHost( T& val );

    // copies count() elements from host --> device queue
    // this is automatically resized to src.size()
    // the head of the host-side src queue is src[0]
    void copyFromHost( const std::vector< T >& src );

    // copies count() elements from device queue --> host
    // dst is automatically resized and the head of the queue is first
    void copyToHost( std::vector< T >& dst ) const;

private:

    // stores an absolute head and tail pointer
    DeviceVariable< uint2 > md_headTailAbsoluteIndices;
    DeviceArray1D< T > md_ringBuffer;

};

#include "DeviceQueue.inl"
