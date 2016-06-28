#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

// A simple frames-per-second estimator based on sampling the last N frame
// times. Construct the class with N and call update() every frame. You may
// query the frame period in milliseconds or frequency in Hz.

class FPSEstimator
{
public:

    FPSEstimator( int nSamples = 128 );

    void update();

    float framePeriodMilliseconds() const;
    float framesPerSecond() const;

    // Returns the average frame period, rounded to the nearest millisecond, as
    // a string.
    std::string framePeriodMillisecondsString() const;

    // Returns the average framerate, rounded to the nearest Hz, as a string.
    std::string framesPerSecondString() const;

private:

    using Clock = std::chrono::high_resolution_clock;
    using FloatDurationMS = std::chrono::duration< float, std::milli >;

    Clock::time_point m_lastUpdateTime;

    bool m_isFirstUpdate = true;
    int m_nextSampleIndex = 0;
    int m_nActualSamples = 0;
    std::vector< Clock::duration > m_frameTimeSamples;

};
