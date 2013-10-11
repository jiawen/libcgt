#include "time/PerformanceCollector.h"

PerformanceCollector::PerformanceCollector()
{
  m_clock.start();
}

void PerformanceCollector::registerEvent( QString name )
{
	m_eventStartTimes[ name ] = 0;
	m_eventTotalElapsedTime[ name ] = 0;
	m_eventCounts[ name ] = 0;
}

void PerformanceCollector::unregisterEvent( QString name )
{
	m_eventStartTimes.remove( name );
	m_eventTotalElapsedTime.remove( name );
	m_eventCounts.remove( name );
}

void PerformanceCollector::beginEvent( QString name )
{
	qint64 now = m_clock.elapsed();
	m_eventStartTimes[ name ] = now;
}

void PerformanceCollector::endEvent( QString name )
{
	qint64 now = m_clock.elapsed();
	qint64 last = m_eventStartTimes[ name ];
	qint64 dt = now - last;

	m_eventTotalElapsedTime[ name ] += dt;
	++m_eventCounts[ name ];
}

float PerformanceCollector::averageTimeMilliseconds( QString name )
{
	qint64 dt = m_eventTotalElapsedTime[ name ];
	int count = m_eventCounts[ name ];
	float dtMillis = static_cast< float >( dt );
	return dtMillis / count;
}
