#pragma once

#include <vector>
#include <time/Clock.h>

// A simple frames-per-second estimator
// based on sampling the last N frame times
// Construct the class with N
// and call update() every frame
// fps() will return the fps as a float

class FPSEstimator
{
public:

	FPSEstimator( int nSamples );

	float framePeriodMilliseconds() const;
	float framesPerSecond() const;

	void update();

private:

	Clock m_clock;
	
	bool m_isFirstUpdate;
	int64 m_lastUpdateTime;
	int m_nextSampleIndex;
	int m_nActualSamples;
	std::vector< int64 > m_frameTimeSamples;

};
