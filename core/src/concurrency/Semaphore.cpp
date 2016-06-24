#include "Semaphore.h"

namespace libcgt { namespace core { namespace concurrency {

Semaphore::Semaphore( int count ) : m_count( count )
{

}

void Semaphore::signal( int n )
{
    std::unique_lock< std::mutex > lock( m_mutex );
    ++m_count;
    m_cv.notify_one();
}

void Semaphore::wait( int n )
{
    std::unique_lock< std::mutex > lock( m_mutex );
    m_cv.wait
    (
        lock,
        [&]
        {
            return ( m_count >= n );
        }
    );
    m_count -= n;
}

bool Semaphore::tryWait( int n )
{
    std::unique_lock< std::mutex > lock( m_mutex );
    if( m_count < n )
    {
        return false;
    }
    m_count -= n;
    return true;
}

bool Semaphore::tryWait( int n, int milliseconds )
{
    std::unique_lock< std::mutex > lock( m_mutex );
    bool acquired = m_cv.wait_for
    (
        lock, std::chrono::milliseconds( milliseconds ),
        [&]
        {
            return ( m_count >= n );
        }
    );
    return acquired;
}

int Semaphore::count() const
{
    return m_count;
}

} } } // concurrency, core, libcgt
