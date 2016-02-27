#pragma once

#include <vector>

#include "Semaphore.h"

// A concurrent bounded FIFO queue for ONE producer thread and ONE consumer
// thread only.

template< typename T >
class ConcurrentQueue
{
public:

    ConcurrentQueue( int nItems );

    int bufferSize() const;

    // Blocks the producer until an entry is available then copies the item into
    // the ring buffer.
    void enqueue( const T& item );

    // Block the producer until an entry is available, then returns a pointer to
    // the entry. The producer must call endEnqueue() to guarantee correctness.
    T* beginEnqueue();
    void endEnqueue();

    // Blocks the consumer until an entry is available, then returns a copy of
    // the item.
    T dequeue();

    // Blocks the consumer until an entry is available, then returns a pointer
    // to the entry. The consumer must call endDequeue() to guarantee
    // correctness.
    T* beginDequeue();
    void endDequeue();

    // Returns true if there is at least one slot available for reading.
    bool availableForReading() const;

    // Returns the entries available to be read.
    int numEntriesFilled() const;

    // Returns true if there is at least one slot available for writing.
    bool availableForWriting() const;

    // Returns the number of slots available for writing.
    int numEntriesFree() const;

private:

    std::vector< T > m_buffer;

    // # slots available for the producer to write to.
    Semaphore m_nSlotsFree;
    // # slots written to by producer and not yet read by consumer
    Semaphore m_nSlotsFilled;

    int m_headIndex; // Where to read from.
    int m_tailIndex; // Where to write to.
};

template< typename T >
ConcurrentQueue< T >::ConcurrentQueue( int nItems ) :

    m_nSlotsFree( nItems ),
    m_nSlotsFilled( 0 ),
    m_buffer( nItems ),

    m_headIndex( 0 ),
    m_tailIndex( 0 )
{

}

template< typename T >
int ConcurrentQueue< T >::bufferSize() const
{
    return static_cast< int >( m_buffer.size() );
}

template< typename T >
void ConcurrentQueue< T >::enqueue( const T& item )
{
    T* pEntry = beginEnqueue();
    *pEntry = item;
    endEnqueue();
}

template< typename T >
T* ConcurrentQueue< T >::beginEnqueue()
{
    m_nSlotsFree.wait();
    return &( m_buffer[ m_tailIndex ] );
}

template< typename T >
void ConcurrentQueue< T >::endEnqueue()
{
    m_tailIndex = ( m_tailIndex + 1 ) % bufferSize();
    m_nSlotsFilled.signal();
}

template< typename T >
T ConcurrentQueue< T >::dequeue()
{
    T* pEntry = beginDequeue();
    T item = *pEntry;
    endDequeue();
    return item;
}

template< typename T >
T* ConcurrentQueue< T >::beginDequeue()
{
    m_nSlotsFilled.wait();
    return &( m_buffer[ m_headIndex ] );
}

template< typename T >
void ConcurrentQueue< T >::endDequeue()
{
    m_headIndex = ( m_headIndex + 1 ) % bufferSize();
    m_nSlotsFree.signal();
}

template< typename T >
bool ConcurrentQueue< T >::availableForReading() const
{
    return( numEntriesFilled() > 0 );
}

template< typename T >
int ConcurrentQueue< T >::numEntriesFilled() const
{
    return m_nSlotsFilled.available();
}

template< typename T >
bool ConcurrentQueue< T >::availableForWriting() const
{
    return( numEntriesFree() > 0 );
}

template< typename T >
int ConcurrentQueue< T >::numEntriesFree() const
{
    return m_nSlotsFree.available();
}
