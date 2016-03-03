template< typename T >
__inline__ __host__ __device__
KernelQueue< T >::KernelQueue() :

    md_pHeadTailAbsoluteIndices( nullptr )

{

}

template< typename T >
__inline__ __host__ __device__
KernelQueue< T >::KernelQueue( uint2* d_pHeadTailAbsoluteIndices, KernelVector< T > ringBuffer ) :

    md_pHeadTailAbsoluteIndices( d_pHeadTailAbsoluteIndices ),
    md_ringBuffer( ringBuffer )
{

}

#ifdef __CUDACC__

template< typename T >
__inline__ __device__
uint KernelQueue< T >::enqueue( const T& val )
{
    uint absoluteTailIndex = atomicAdd( tailIndexPointer(), 1u );
    uint tailIndex = absoluteTailIndex % capacity();
    md_ringBuffer[ tailIndex ] = val;
    return tailIndex;
}

template< typename T >
__inline__ __device__
T KernelQueue< T >::dequeue()
{
    uint absoluteHeadIndex = atomicAdd( headIndexPointer(), 1u );
    uint headIndex = absoluteHeadIndex % capacity();
    return md_ringBuffer[ headIndex ];
}

template< typename T >
__inline__ __device__
T* KernelQueue< T >::enqueueN( uint n )
{
    uint absoluteTailIndex = atomicAdd( tailIndexPointer(), n );
    uint tailIndex = absoluteTailIndex % capacity();
    return md_ringBuffer.pointer + tailIndex;
}

template< typename T >
__inline__ __device__
T* KernelQueue< T >::dequeueN( uint n )
{
    uint absoluteHeadIndex = atomicAdd( headIndexPointer(), n );
    uint headIndex = absoluteHeadIndex % capacity();
    return md_ringBuffer.pointer + headIndex;
}

#endif

template< typename T >
__inline__ __device__
int KernelQueue< T >::count()
{
    return( md_pHeadTailAbsoluteIndices->y - md_pHeadTailAbsoluteIndices->x );
}

template< typename T >
__inline__ __device__
int KernelQueue< T >::capacity() const
{
    return md_ringBuffer.length;
}

template< typename T >
__inline__ __device__
bool KernelQueue< T >::isEmpty()
{
    int absoluteHeadIndex = *( headIndexPointer() );
    int absoluteTailIndex = *( tailIndexPointer() );
    return( absoluteHeadIndex == absoluteTailIndex );
}

template< typename T >
__inline__ __device__
bool KernelQueue< T >::isFull()
{
    // we can distinguish between full and empty
    // because we use absolute indices:
    //
    // if we ever wrote anything to the queue: then check if the count() is less than capacity()
    // if they're equal, then it's full
    int absoluteTailIndex = *( tailIndexPointer() );
    if( absoluteTailIndex > 0 )
    {
        return( count() < capacity() );
    }
    // if we never wrote anything to the queue, then it clearly can't be full
    else
    {
        return false;
    }
}

template< typename T >
__inline__ __device__
KernelVector< T >& KernelQueue< T >::ringBuffer()
{
    return md_ringBuffer;
}

template< typename T >
__inline__ __device__
uint* KernelQueue< T >::headIndexPointer()
{
    return &( md_pHeadTailAbsoluteIndices->x );
}

template< typename T >
__inline__ __device__
uint* KernelQueue< T >::tailIndexPointer()
{
    return &( md_pHeadTailAbsoluteIndices->y );
}

#if 0
    // tries to enqueue a value
    // returns false if the queue is full
    __inline__ __device__
        bool tryEnqueue( const T& val )
    {
        // use some magic trick with atomicInc with elements.length?
    }

    // tries to dequeue a value
    // returns false if
    __inline__ __device__
        bool tryDequeue( T& val )
    {
        uint tail = *( tailPointer() );
        // atomicInc computes:
        // uint oldHead = *headPointer;
        // if( oldHead >= tail )
        // {
        //     return 0;
        // }
        // else
        // {
        //     *headPointer = oldHead + 1
        //     return oldHead;
        // }
        uint retVal = atomicInc( headPointer(), tail );
        return ( retVal != 0 );
    }
#endif
