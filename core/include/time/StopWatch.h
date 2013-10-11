#pragma once

#if 0

#include "Clock.h"

class StopWatch
{
public:

	// Creates a StopWatch and resets it
	StopWatch();

	// Resets the start time to now
	// and returns millisecondsElapsed()
	float reset();

	// Returns the number of milliseconds since the start time
	float millisecondsElapsed();	
	
private:

	Clock m_clock;
	int64 m_resetTime;

};
#endif
