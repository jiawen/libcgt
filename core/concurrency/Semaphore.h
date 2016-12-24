#pragma once

#include <condition_variable>
#include <mutex>

namespace libcgt { namespace core { namespace concurrency {

// A simple semaphore implementation based on a mutex and a condition variable.
// Recall that unlike mutexes (which are lockable objects), semaphores simply
// keep track of shared resources and do not have ownership semantics. Any
// thread may wait() to acquire a resource (decrementing the counter), or
// signal() to release a resource (incrementing the counter).
class Semaphore
{
public:

    Semaphore( int count = 0 );

    // Atomically increment the counter by n (indicating that n resources are
    // are now available). Then wake up *one* thread that may be waiting.
    void signal( int n = 1 );

    // Attempt to decrement the counter by n (acquiring n resources).
    // If less than n resources are available, then block the thread until at
    // least n resources are available.
    void wait( int n = 1 );

    // Attempt to decrement the counter by n (acquiring n resources), returning
    // immediately.
    //
    // Returns true if the client succeeded.
    // If less than n resources are available, returns false.
    bool tryWait( int n );

    // Attempt to decrement the counter by n (acquiring n resources).
    //
    // Returns true if the client succeeded.
    //
    // If less than n resources are available, then block the thread until at
    // least n resources are available, or until "milliseconds" milliseconds
    // has expired. Returns true if resources were acquired.
    //
    // If milliseconds < 0, will be equivalent to tryWait( n ).
    bool tryWait( int n, int milliseconds );

    // Return the current number of resources available.
    int count() const;

private:

    std::mutex m_mutex;
    std::condition_variable m_cv;
    int m_count;
};

} } } // concurrency, core, libcgt
