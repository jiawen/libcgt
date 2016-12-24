#include "libcgt/core/time/PerformanceCollector.h"

void PerformanceCollector::registerEvent( const std::string& name )
{
    m_eventStartTime[ name ] = Clock::now();
    m_eventTotalElapsedTime[ name ] = Clock::duration::zero();
    m_eventCounts[ name ] = 0;
}

void PerformanceCollector::unregisterEvent( const std::string& name )
{
    m_eventStartTime.erase( name );
    m_eventTotalElapsedTime.erase( name );
    m_eventCounts.erase( name );
}

void PerformanceCollector::beginEvent( const std::string& name )
{
    m_eventStartTime[ name ] = PerformanceCollector::Clock::now();
}

void PerformanceCollector::endEvent( const std::string& name )
{
    auto now = PerformanceCollector::Clock::now();
    auto last = m_eventStartTime[ name ];
    auto dt = now - last;

    m_eventTotalElapsedTime[ name ] += dt;
    ++m_eventCounts[ name ];
}

float PerformanceCollector::averageTimeMilliseconds( const std::string& name )
{
    if( m_eventCounts.find( name ) == m_eventCounts.end() ||
       m_eventCounts[ name ] == 0 )
    {
        return 0.0f;
    }
    auto dt = m_eventTotalElapsedTime[ name ];
    int count = m_eventCounts[ name ];
    float dtMillis = std::chrono::duration_cast<
        PerformanceCollector::FloatDurationMS >( dt ).count();
    return dtMillis / count;
}
