#pragma once

#include <vector>
#include <QSemaphore>

// An atomic FIFO queue that guarantees atomicity
// for ONE producer thread and ONE consumer thread

template< typename T >
class QAtomicQueue
{
public:

    QAtomicQueue( int nItems );

    int bufferSize() const;

    // blocks the producer until an entry is available
    // then copies the item
    void enqueue( const T& item );

    // blocks the producer until an entry is available
    // then returns a pointer to the entry
    // producer must call endEnqueue() to guarantee correctness
    T* beginEnqueue();
    void endEnqueue();

    // blocks the consumer until an entry is available
    // then copies the item
    T dequeue();

    // blocks the consumer until an entry is available
    // then returns a pointer to the entry
    // consumer must call endDequeue() to guarantee correctness
    T* beginDequeue();
    void endDequeue();

    bool availableForReading() const;

    // returns the entries available to be read
    int numEntriesFilled() const;

    bool availableForWriting() const;

    // returns the entries available to be written
    int numEntriesFree() const;

private:

    std::vector< T > m_buffer;

    QSemaphore m_nSlotsFree; // # slots available for the producer to write to
    QSemaphore m_nSlotsFilled; // # slots written to by producer and not yet read by consumer

    int m_headIndex; // where to read from
    int m_tailIndex; // where to write to
};

template< typename T >
QAtomicQueue< T >::QAtomicQueue( int nItems ) :

    m_nSlotsFree( nItems ),
    m_nSlotsFilled( 0 ),
    m_buffer( nItems ),

    m_headIndex( 0 ),
    m_tailIndex( 0 )
{

}

template< typename T >
int QAtomicQueue< T >::bufferSize() const
{
    return static_cast< int >( m_buffer.size() );
}

template< typename T >
void QAtomicQueue< T >::enqueue( const T& item )
{
    T* pEntry = beginEnqueue();
    *pEntry = item;
    endEnqueue();
}

template< typename T >
T* QAtomicQueue< T >::beginEnqueue()
{
    m_nSlotsFree.acquire();
    return &( m_buffer[ m_tailIndex ] );
}

template< typename T >
void QAtomicQueue< T >::endEnqueue()
{
    m_tailIndex = ( m_tailIndex + 1 ) % bufferSize();
    m_nSlotsFilled.release();
}

template< typename T >
T QAtomicQueue< T >::dequeue()
{
    T* pEntry = beginDequeue();
    T item = *pEntry;
    endDequeue();
    return item;
}

template< typename T >
T* QAtomicQueue< T >::beginDequeue()
{
    m_nSlotsFilled.acquire();
    return &( m_buffer[ m_headIndex ] );
}

template< typename T >
void QAtomicQueue< T >::endDequeue()
{
    m_headIndex = ( m_headIndex + 1 ) % bufferSize();
    m_nSlotsFree.release();
}

template< typename T >
bool QAtomicQueue< T >::availableForReading() const
{
    return( numEntriesFilled() > 0 );
}

template< typename T >
int QAtomicQueue< T >::numEntriesFilled() const
{
    return m_nSlotsFilled.available();
}

template< typename T >
bool QAtomicQueue< T >::availableForWriting() const
{
    return( numEntriesFree() > 0 );
}

template< typename T >
int QAtomicQueue< T >::numEntriesFree() const
{
    return m_nSlotsFree.available();
}
