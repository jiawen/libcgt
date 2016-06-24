namespace libcgt { namespace core { namespace concurrency {

template< typename T >
BoundedConcurrentQueue< T >::BoundedConcurrentQueue( int bufferSize,
    const T& fill ) :
    m_nSlotsFree( bufferSize ),
    m_nSlotsFilled( 0 ),
    m_buffer( bufferSize, fill ),

    m_headIndex( 0 ),
    m_tailIndex( 0 )
{

}

template< typename T >
BoundedConcurrentQueue< T >::BoundedConcurrentQueue(
    std::vector< T >&& ringBuffer ) :
    m_nSlotsFree( static_cast< int >( ringBuffer.size() ) ),
    m_nSlotsFilled( 0 ),
    m_buffer( std::move( ringBuffer ) ),

    m_headIndex( 0 ),
    m_tailIndex( 0 )
{

}

template< typename T >
size_t BoundedConcurrentQueue< T >::bufferSize() const
{
    return m_buffer.size();
}

template< typename T >
void BoundedConcurrentQueue< T >::enqueue( const T& item )
{
    T* pEntry = beginEnqueue();
    *pEntry = item;
    endEnqueue();
}

template< typename T >
bool BoundedConcurrentQueue< T >::tryEnqueue( const T& item, int milliseconds )
{
    T* pEntry = tryBeginEnqueue( milliseconds );
    bool success = ( pEntry != nullptr );
    if( success )
    {
        *pEntry = item;
        endEnqueue();
    }
    return success;
}

template< typename T >
T* BoundedConcurrentQueue< T >::beginEnqueue()
{
    m_nSlotsFree.wait( 1 );
    return &( m_buffer[ m_tailIndex ] );
}

template< typename T >
void BoundedConcurrentQueue< T >::endEnqueue()
{
    m_tailIndex = ( m_tailIndex + 1 ) % bufferSize();
    m_nSlotsFilled.signal();
}

template< typename T >
T* BoundedConcurrentQueue< T >::tryBeginEnqueue( int milliseconds )
{
    bool succeeded = m_nSlotsFree.tryWait( 1, milliseconds );
    if( succeeded )
    {
        return &( m_buffer[ m_tailIndex ] );
    }
    else
    {
        return nullptr;
    }
}

template< typename T >
T BoundedConcurrentQueue< T >::dequeue()
{
    T* pEntry = beginDequeue();
    T item = *pEntry;
    endDequeue();
    return item;
}

template< typename T >
bool BoundedConcurrentQueue< T >::tryDequeue( T& output, int milliseconds )
{
    T* pEntry = tryBeginDequeue( milliseconds );
    bool succeeded = ( pEntry != nullptr );
    if( succeeded )
    {
        output = *pEntry;
        endDequeue();
    }
    return succeeded;
}

template< typename T >
T* BoundedConcurrentQueue< T >::beginDequeue()
{
    m_nSlotsFilled.wait();
    return &( m_buffer[ m_headIndex ] );
}

template< typename T >
T* BoundedConcurrentQueue< T >::tryBeginDequeue( int milliseconds )
{
    bool succeeded = m_nSlotsFilled.tryWait( 1, milliseconds );
    if( succeeded )
    {
        return &( m_buffer[ m_headIndex ] );
    }
    else
    {
        return nullptr;
    }
}

template< typename T >
void BoundedConcurrentQueue< T >::endDequeue()
{
    m_headIndex = ( m_headIndex + 1 ) % bufferSize();
    m_nSlotsFree.signal();
}

template< typename T >
bool BoundedConcurrentQueue< T >::availableForReading() const
{
    return( numEntriesFilled() > 0 );
}

template< typename T >
int BoundedConcurrentQueue< T >::numEntriesFilled() const
{
    return m_nSlotsFilled.count();
}

template< typename T >
bool BoundedConcurrentQueue< T >::availableForWriting() const
{
    return( numEntriesFree() > 0 );
}

template< typename T >
int BoundedConcurrentQueue< T >::numEntriesFree() const
{
    return m_nSlotsFree.count();
}

} } } // concurrency, core, libcgt
