#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>

class PerformanceCollector
{
public:

    PerformanceCollector() = default;

    // Register an event for performance collection.
    void registerEvent( const std::string& name );

    // Unregister an event from performance collection.
    void unregisterEvent( const std::string& name );

    // Resets the statistics for an event.
    void resetEvent( const std::string& name );

    // Call each time an event starts.
    void beginEvent( const std::string& name );

    // Call each time an event ends.
    void endEvent( const std::string& name );

    // Returns the average time spent on an event over all
    // beginEvent()/endEvent pairs.
    float averageTimeMilliseconds( const std::string& name );

private:

    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration< float, std::milli > FloatDurationMS;

    // In nanoseconds.
    std::unordered_map< std::string, Clock::time_point > m_eventStartTime;
    std::unordered_map< std::string, Clock::duration > m_eventTotalElapsedTime;
    std::unordered_map< std::string, int > m_eventCounts;

};
