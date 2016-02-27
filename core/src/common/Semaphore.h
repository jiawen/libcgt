#pragma once

#include <condition_variable>
#include <mutex>

// A simple semaphore implementation based on a mutex and a condition variable.
// Recall that unlike mutexes (which are lockable objects), semaphores simply
// keep track of shared resources and do not have ownership semantics. Any
// thread may wait() to acquire a resource (decrementing the counter), or
// signal() to release a resources (incrementing the counter).
class Semaphore
{
public:

    Semaphore( int count = 0 );

    // Atomically increment the counter, then wake up *one* thread that may be
    // waiting.
    void signal();

    // Attempt to decrement the counter. If it is already zero, then block the
    // thread until the semaphore is signaled.
    void wait();

private:

    std::mutex m_mutex;
    std::condition_variable m_cv;
    int m_count;
};