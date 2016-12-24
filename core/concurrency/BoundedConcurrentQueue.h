#pragma once

#include <vector>

#include "Semaphore.h"

namespace libcgt { namespace core { namespace concurrency {

// A bounded concurrent FIFO queue for ONE producer thread and ONE consumer
// thread only. The queue is backed by a fixed-length array and expects the
// producer and consumer to "check out" subsets of the the underlying buffer
// and fill or read from them.
//
// TODO(jiawen): make another type of queue, where you own the buffers and send
// them around.
//
// TODO(jiawen): relax the single consumer assumption.
template< typename T >
class BoundedConcurrentQueue
{
public:

    // Create a BoundedConcurrentQueue with a fixed size "bufferSize" and each
    // filled with "fill".
    BoundedConcurrentQueue( int bufferSize, const T& fill = T() );

    // Create a BoundedConcurrentQueue with a pre-initialized fixed-sized
    // backing store called "ringBuffer".
    BoundedConcurrentQueue( std::vector< T >&& ringBuffer );

    size_t bufferSize() const;

    // Blocks the producer until an entry is available then copies the item
    // into the ring buffer.
    void enqueue( const T& item );

    // Attempt to enqueue an item, waiting up to "milliseconds" milliseconds.
    // Returns whether it succeeded.
    bool tryEnqueue( const T& item, int milliseconds );

    // TODO(jiawen): beginEnqueue( int n ). Return a view object that
    // automatically calls endEnqueue().
    // Block the producer until an entry is available, then returns a pointer
    // to the entry.
    //
    // The producer must call endEnqueue() to guarantee correctness.
    T* beginEnqueue();

    // Attempt to acquire an entry into which the producer can write, waiting
    // up to "milliseconds" milliseconds. Returns a pointer to tne entry on
    // success, and nullptr on timeout.
    //
    // On success, the producer must call endEnqueue() to guarantee correctness.
    T* tryBeginEnqueue( int milliseconds );

    void endEnqueue();

    // Blocks the consumer until an entry is available, then returns a copy of
    // the item.
    T dequeue();

    // Attempt to dequeue an item, waiting up to "milliseconds" milliseconds.
    // On success, writes the output into "output.
    //
    // Returns whether the operation succeeded.
    bool tryDequeue( T& output, int milliseconds );

    // Blocks the consumer until an entry is available, then returns a pointer
    // to the entry. The consumer must call endDequeue() to guarantee
    // correctness.
    T* beginDequeue();

    // Attempt to acquire an entry from which the consumer can read, waiting
    // up to "milliseconds" milliseconds. Returns a pointer to tne entry on
    // success, and nullptr on timeout.
    //
    // On success, the consumer must call endDequeue() to guarantee correctness.
    T* tryBeginDequeue( int milliseconds );

    void endDequeue();

    // Returns true if there is at least one slot available for reading.
    bool availableForReading() const;

    // Returns the entries available for reading.
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

} } } // concurrency, core, libcgt

#include "BoundedConcurrentQueue.inl"
