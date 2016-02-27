#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#include "math/Arithmetic.h"

// TODO: use a mutex on notifications to make the parallel version work

class ProgressReporter
{
public:

    // Construct a new ProgressReporter with the prefix string "Working:"
    // a predetermined number of tasks, and a reportRate of 1%
    ProgressReporter( int nTasks );

    // Construct a new ProgressReporter given a prefix string,
    // a predetermined number of tasks, and a reportRate of 1%
    ProgressReporter( const std::string& prefix, int nTasks );

    ProgressReporter( const std::string& prefix, int nTasks,
                     float reportRatePercent );

    std::string notifyAndGetProgressString();
    void notifyAndPrintProgressString();
    void notifyTaskCompleted();

    std::string getProgressString();

    float percentComplete();
    bool isComplete();
    int numTasksRemaining();
    float approximateMillisecondsRemaining();
    float averageMillisecondsPerTask();

private:

    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::time_point< Clock > TimePoint;

    void initialize( const std::string& prefix, int nTasks,
                    float reportRatePercent );

    std::string m_prefix;
    int m_nTasks;
    float m_reportRatePercent;

    TimePoint m_startTime;
    TimePoint m_previousTaskCompletedTime;

    int64_t m_totalMillisecondsElapsed = 0;
    float m_nextReportedPercent = 0;
    int m_nTasksCompleted = 0;
};
