#include "libcgt/core/common/ProgressReporter.h"

ProgressReporter::ProgressReporter( int nTasks )
{
    initialize( "Working:", nTasks, 1 );
}

ProgressReporter::ProgressReporter( const std::string& prefix, int nTasks )
{
    initialize( prefix, nTasks, 1 );
}

ProgressReporter::ProgressReporter( const std::string& prefix, int nTasks,
                                   float reportRatePercent )
{
    initialize( prefix, nTasks, reportRatePercent );
}

std::string ProgressReporter::notifyAndGetProgressString()
{
    notifyTaskCompleted();
    return getProgressString();
}

void ProgressReporter::notifyAndPrintProgressString()
{
    notifyTaskCompleted();
    if( m_reportRatePercent < 0 )
    {
        printf( "%s\n", getProgressString().c_str() );
    }
    else if( isComplete() || percentComplete() > m_nextReportedPercent )
    {
        printf( "%s\n", getProgressString().c_str() );
        m_nextReportedPercent += m_reportRatePercent;
    }
}

void ProgressReporter::notifyTaskCompleted()
{
    auto now = ProgressReporter::Clock::now();
    int64_t millisecondsForTask =
        std::chrono::duration_cast< std::chrono::milliseconds >(
            now - m_previousTaskCompletedTime ).count();
    m_previousTaskCompletedTime = now;

    m_totalMillisecondsElapsed += millisecondsForTask;

    if( m_nTasksCompleted < m_nTasks )
    {
        ++m_nTasksCompleted;
    }
}

std::string ProgressReporter::getProgressString()
{
    if( numTasksRemaining() <= 0 )
    {
        return m_prefix + " 100% [done!]";
    }
    else
    {
        std::string timeRemainingString;
        if( approximateMillisecondsRemaining() < 1000 )
        {
            int ms = libcgt::core::math::roundToInt
                ( approximateMillisecondsRemaining() );
            timeRemainingString = std::to_string( ms ) + " ms";
        }
        else
        {
            float sec = approximateMillisecondsRemaining() / 1000.0f;
            timeRemainingString = std::to_string( sec ) + " s";
        }

        std::string timeElapsedString;
        if( m_totalMillisecondsElapsed < 1000 )
        {
            timeElapsedString = std::to_string( m_totalMillisecondsElapsed )
                + " ms";
        }
        else
        {
            timeElapsedString =
                std::to_string( m_totalMillisecondsElapsed / 1000.0f ) + " s";
        }

        return m_prefix
            + std::to_string( percentComplete() ) + "% "
            + std::to_string( numTasksRemaining() ) + " tasks left "
            + "(" + timeRemainingString + "), elapsed: "
            + timeElapsedString;
    }
}

float ProgressReporter::percentComplete()
{
    return 100.0f *
        libcgt::core::math::divideIntsToFloat( m_nTasksCompleted, m_nTasks );
}

bool ProgressReporter::isComplete()
{
    return ( m_nTasksCompleted == m_nTasks );
}

int ProgressReporter::numTasksRemaining()
{
    return m_nTasks - m_nTasksCompleted;
}

float ProgressReporter::approximateMillisecondsRemaining()
{
    return numTasksRemaining() * averageMillisecondsPerTask();
}

float ProgressReporter::averageMillisecondsPerTask()
{
    return static_cast< float >( m_totalMillisecondsElapsed ) / m_nTasksCompleted;
}

void ProgressReporter::initialize( const std::string& prefix, int nTasks,
                                  float reportRatePercent )
{
    if( prefix.back() == ':' )
    {
        m_prefix = prefix;
    }
    else
    {
        m_prefix = prefix + ":";
    }

    m_nTasks = nTasks;
    m_reportRatePercent = reportRatePercent;

    m_startTime = ProgressReporter::Clock::now();
    m_previousTaskCompletedTime = m_startTime;
}
