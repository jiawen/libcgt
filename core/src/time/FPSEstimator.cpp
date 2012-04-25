#include "time/FPSEstimator.h"

#include <math/MathUtils.h>

FPSEstimator::FPSEstimator( int nSamples ) :

	m_isFirstUpdate( true ),
	m_nextSampleIndex( 0 ),
	m_nActualSamples( 0 ),
	m_frameTimeSamples( nSamples, 0 )

{

}

void FPSEstimator::update()
{
	if( m_isFirstUpdate )
	{
		m_lastUpdateTime = m_clock.getCounterValue();
		m_isFirstUpdate = false;
	}
	else
	{
		int64 now = m_clock.getCounterValue();
		int64 dt = now - m_lastUpdateTime;

		int n = m_frameTimeSamples.size();
		m_nActualSamples = MathUtils::clampToRangeInt( m_nActualSamples + 1, 0, n + 1 );

		m_frameTimeSamples[ m_nextSampleIndex ] = dt;

		m_nextSampleIndex = ( m_nextSampleIndex + 1 ) % n;

		m_lastUpdateTime = m_clock.getCounterValue();
	}	
}

float FPSEstimator::framePeriodMilliseconds() const
{
	if( m_nActualSamples > 0 )
	{
		int64 sum = 0;
		for( int i = 0; i < m_nActualSamples; ++i )
		{
			sum += m_frameTimeSamples[ i ];
		}
		return( m_clock.convertIntervalToMillis( sum ) / m_nActualSamples );
	}
	else
	{
		return 0;
	}
}

float FPSEstimator::framesPerSecond() const
{
	if( m_nActualSamples > 0 )
	{		
		return 1000.0f / framePeriodMilliseconds();
	}
	else
	{
		return 0;
	}
}

QString FPSEstimator::framePeriodMillisecondsString() const
{
	float dt = framePeriodMilliseconds();
	return QString( "%1" ).arg( dt, 0, 'f', 1 );
}

QString FPSEstimator::framesPerSecondString() const
{
	float fps = framesPerSecond();
	return QString( "%1" ).arg( fps, 0, 'f', 1 );
}
