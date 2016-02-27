#include "time/FPSEstimator.h"

#include <math/MathUtils.h>

FPSEstimator::FPSEstimator( int nSamples ) :
    m_frameTimeSamples( nSamples )
{

}

void FPSEstimator::update()
{
    if( m_isFirstUpdate )
    {
        m_lastUpdateTime = Clock::now();
        m_isFirstUpdate = false;
    }
    else
    {
        auto now = Clock::now();
        auto dt = now - m_lastUpdateTime;

        int n = static_cast< int >( m_frameTimeSamples.size() );
        m_nActualSamples = MathUtils::clampToRangeExclusive(
            m_nActualSamples + 1, 0, n + 1 );

        m_frameTimeSamples[ m_nextSampleIndex ] = dt;

        m_nextSampleIndex = ( m_nextSampleIndex + 1 ) % n;

        m_lastUpdateTime = now;
    }
}

float FPSEstimator::framePeriodMilliseconds() const
{
    if( m_nActualSamples > 0 )
    {
        auto sum = Clock::duration::zero();
        for( int i = 0; i < m_nActualSamples; ++i )
        {
            sum += m_frameTimeSamples[ i ];
        }
        float sumMS = std::chrono::duration_cast<
            FPSEstimator::FloatDurationMS >( sum ).count();
        return( sumMS / m_nActualSamples );
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

std::string FPSEstimator::framePeriodMillisecondsString() const
{
    return std::to_string( framePeriodMilliseconds() );
}

std::string FPSEstimator::framesPerSecondString() const
{
    return std::to_string( framesPerSecond() );
}
