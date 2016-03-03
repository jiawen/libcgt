#include "ErrorChecking.h"

template< typename T >
DeviceQueue< T >::DeviceQueue()
{

}

template< typename T >
DeviceQueue< T >::DeviceQueue( int capacity )
{
    resize( capacity );
}

template< typename T >
DeviceQueue< T >::DeviceQueue( const DeviceQueue< T >& copy ) :

    md_headTailAbsoluteIndices( copy.md_headTailAbsoluteIndices ),
    md_ringBuffer( copy.md_ringBuffer )

{

}

template< typename T >
DeviceQueue< T >::DeviceQueue( DeviceQueue< T >&& move ) :

    md_headTailAbsoluteIndices( std::move( move.md_headTailAbsoluteIndices ) ),
    md_ringBuffer( std::move( move.md_ringBuffer ) )

{

}

template< typename T >
DeviceQueue< T >& DeviceQueue< T >::operator = ( const DeviceQueue< T >& copy )
{
    if( this != &copy )
    {
        md_headTailAbsoluteIndices = copy.md_headTailAbsoluteIndices;
        md_ringBuffer = copy.md_ringBuffer;
    }
    return *this;
}

template< typename T >
DeviceQueue< T >& DeviceQueue< T >::operator = ( DeviceQueue< T >&& move )
{
    if( this != &move )
    {
        md_headTailAbsoluteIndices = std::move( move.md_headTailAbsoluteIndices );
        md_ringBuffer = std::move( move.md_ringBuffer );
    }
    return *this;
}

template< typename T >
bool DeviceQueue< T >::isNull() const
{
    return md_ringBuffer.isNull();
}

template< typename T >
bool DeviceQueue< T >::notNull() const
{
    return !isNull();
}

template< typename T >
int DeviceQueue< T >::capacity() const
{
    return md_ringBuffer.length();
}

template< typename T >
size_t DeviceQueue< T >::sizeInBytes() const
{
    return md_ringBuffer.sizeInBytes() + sizeof( uint2 );
}

template< typename T >
void DeviceQueue< T >::resize( int capacity )
{
    md_ringBuffer.resize( capacity );
    clear();
}

template< typename T >
void DeviceQueue< T >::clear( int headIndex, int tailIndex )
{
    uint2 ht = make_uint2( headIndex, tailIndex );
    md_headTailAbsoluteIndices.set( ht );
}

template< typename T >
bool DeviceQueue< T >::isEmpty() const
{
    uint2 ht = md_headTailAbsoluteIndices.get();
    return( ht.x == ht.y );
}

template< typename T >
bool DeviceQueue< T >::isFull() const
{
    // we can distinguish between full and empty
    // because we use absolute indices:
    //
    // if we ever wrote anything to the queue: then check if the count() is less than capacity()
    // if they're equal, then it's full
    uint2 ht = md_headTailAbsoluteIndices.get();
    if( ht.y > 0 )
    {
        int count = ht.y - ht.x;
        return( count < capacity() );
    }
    // if we never wrote anything to the queue, then it clearly can't be full
    else
    {
        return false;
    }
}

template< typename T >
int DeviceQueue< T >::count() const
{
    uint2 ht = md_headTailAbsoluteIndices.get();
    int count = ht.y - ht.x;
    return count;
}

template< typename T >
uint2 DeviceQueue< T >::headAndTailAbsoluteIndices() const
{
    return md_headTailAbsoluteIndices.get();
}

template< typename T >
void DeviceQueue< T >::setHeadAbsoluteIndex( int headIndex )
{
    uint2 ht = md_headTailAbsoluteIndices.get();
    ht.x = headIndex;
    md_headTailAbsoluteIndices.set( ht );
}

template< typename T >
void DeviceQueue< T >::setTailAbsoluteIndex( int tailIndex )
{
    uint2 ht = md_headTailAbsoluteIndices.get();
    ht.y = tailIndex;
    md_headTailAbsoluteIndices.set( ht );
}

template< typename T >
void DeviceQueue< T >::setHeadAndTailAbsoluteIndices( int headIndex, int tailIndex )
{
    setHeadAndTailAbsoluteIndices( make_uint2( headIndex, tailIndex ) );
}

template< typename T >
void DeviceQueue< T >::setHeadAndTailAbsoluteIndices( const uint2& ht )
{
    md_headTailAbsoluteIndices.set( ht );
}

template< typename T >
DeviceVector< T >& DeviceQueue< T >::ringBuffer()
{
    return md_ringBuffer;
}

template< typename T >
KernelQueue< T > DeviceQueue< T >::kernelQueue()
{
    return KernelQueue< T >( md_headTailAbsoluteIndices.devicePointer(), md_ringBuffer.kernelVector() );
}

template< typename T >
bool DeviceQueue< T >::enqueueFromHost( const T& val )
{
    if( !isFull() )
    {
        uint2 ht = md_headTailAbsoluteIndices.get();

        int tailIndex = ht.y % capacity();
        md_ringBuffer.set( tailIndex, val );

        ++ht.y;
        md_headTailAbsoluteIndices.set( ht );

        return true;
    }
    return false;
}

template< typename T >
bool DeviceQueue< T >::dequeueToHost( T& val )
{
    if( !isEmpty() )
    {
        uint2 ht = md_headTailAbsoluteIndices.get();

        int headIndex = ht.x % capacity();
        val = md_ringBuffer.get( headIndex );

        ++ht.x;
        md_headTailAbsoluteIndices.set( ht );

        return true;
    }
    return false;
}

template< typename T >
void DeviceQueue< T >::copyFromHost( const std::vector< T >& src )
{
    uint length = static_cast< uint >( src.size() );
    resize( length ); // resize clears the queue
    md_ringBuffer.copyFromHost( src );
    md_headTailAbsoluteIndices.set( make_uint2( 0, length ) );
}

template< typename T >
void DeviceQueue< T >::copyToHost( std::vector< T >& dst ) const
{
    uint2 ht = md_headTailAbsoluteIndices.get();

    int h = ht.x;
    int t = ht.y;
    int count = t - h;

#if 0
    int elementsLength = md_ringBuffer.length();

    dst.clear();
    dst.reserve( count );
    for( int i = 0; i < count; ++i )
    {
        int ringBufferIndex = ( h + i ) % elementsLength;
        dst.push_back( h_elements[ ringBufferIndex ] );
    }
#endif

    dst.resize( count );

    int hIndex = h % capacity();
    int tIndex = t % capacity();

    // no wraparound: single copy
    if( hIndex < tIndex )
    {
        Array1DView< T > dstView( dst.data(), count );
        md_ringBuffer.copyToHost( dstView, hIndex );
    }
    // wrap around: two copies
    else
    {
        // suppose there was wraparound
        // count = 3, capacity = 5
        // hIndex = 3, tIndex = 1
        // [ 0 1 2 3 4 ]
        //     t   h
        //
        // nElementsHeadToEnd = 5 - 3 = 2
        // output:
        // [ 0 1 2 ]
        //
        // copy md_ringBuffer[ 3, 4 ] to output[ 0, 1 ]
        //
        // count - nElementsHeadToEnd = 3 - 2 = 1
        // copy md_ringBuffer[ 0 ] to output[ 2 ]

        int nElementsHeadToEnd = capacity() - hIndex;
        Array1DView< T > dstHeadToEndView( dst.data(), nElementsHeadToEnd );
        md_ringBuffer.copyToHost( dstHeadToEndView, hIndex );

        Array1DView< T > dstZeroToTailView( dst.data() + nElementsHeadToEnd, count - nElementsHeadToEnd );
        md_ringBuffer.copyToHost( dstZeroToTailView, 0 );
    }
}
