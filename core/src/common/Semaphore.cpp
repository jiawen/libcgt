#include "Semaphore.h"

Semaphore::Semaphore( int count ) : m_count( count )
{

}

void Semaphore::signal()
{
    std::unique_lock< std::mutex > lock( m_mutex );
    ++m_count;
    m_cv.notify_one();
}

void Semaphore::wait()
{
    std::unique_lock< std::mutex > lock( m_mutex );
    while( m_count == 0 )
    {
        m_cv.wait(lock);
    }
    --m_count;
}