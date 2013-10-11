#include "time/Clock.h"

#if 0

#ifdef _WIN32

Clock::Clock()
{
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency( &frequency );
	m_frequency = static_cast< float >( frequency.QuadPart );
}

int64 Clock::getCounterValue() const
{
	LARGE_INTEGER counter;
	QueryPerformanceCounter( &counter );
	return counter.QuadPart;
}

float Clock::getFrequency() const
{
	return m_frequency;
}

float Clock::convertIntervalToMillis( int64 interval ) const
{
	float seconds = ( ( float )interval ) / m_frequency;
	return( seconds * 1000.f );
}

int64 Clock::convertMillisToCounterInterval( float millis ) const
{
	float seconds = millis / 1000.f;
	float counts = seconds * m_frequency;
	return( ( int64 )counts );
}

#else

Clock::Clock()
{
	m_frequency = 1000000.f;
}

int64 Clock::getCounterValue()
{
	timeval timeVal;
	timezone timeZone;

	gettimeofday( &timeVal, &timeZone );

	long int seconds = timeVal.tv_sec;
	long int microseconds = timeVal.tv_usec;

	// return in microseconds
	return( 1000000 * seconds + microseconds );
}

float Clock::getFrequency()
{
	return m_frequency;
}

float Clock::convertIntervalToMillis( int64 interval )
{
	float seconds = ( ( float )interval ) / m_frequency;
	return( seconds * 1000.f );
}

int64 Clock::convertMillisToCounterInterval( float millis )
{
	float seconds = millis / 1000.f;
	float counts = seconds * m_frequency;
	return( ( int64 )counts );
}

#endif

#endif
