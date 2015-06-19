#pragma once

#if 0
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include <common/BasicTypes.h>

// A high-precision clock with 64-bit counts
// Not useful for absolute times (day of year, etc)
// just relative differences
class Clock
{
public:

	Clock();

	// Returns the current time in counts
	int64 getCounterValue() const;

	// Returns the frequency (counts per second)	
	float getFrequency() const;

	// Given a difference in counts (two calls to getCounterValue())
	// Returns the milliseconds between them
	float convertIntervalToMillis( int64 interval ) const;

	// The inverse of convertIntervalToMillis()
	int64 convertMillisToCounterInterval( float millis ) const;

private:

	float m_frequency;
};
#endif
