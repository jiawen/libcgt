#include "time/StopWatch.h"

StopWatch::StopWatch()
{
	reset();
}

float StopWatch::reset()
{
	float elapsed = millisecondsElapsed();

	m_resetTime = m_clock.getCounterValue();

	return elapsed;
}

float StopWatch::millisecondsElapsed()
{
	int64 now = m_clock.getCounterValue();
	return m_clock.convertIntervalToMillis( now - m_resetTime );
}
