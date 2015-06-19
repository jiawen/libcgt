#pragma once

#include <vector>
#include <QElapsedTimer>

class QString;

// A simple frames-per-second estimator
// based on sampling the last N frame times
// Construct the class with N
// and call update() every frame
// fps() will return the fps as a float

class FPSEstimator
{
public:

	FPSEstimator( int nSamples = 128 );

	void update();

	float framePeriodMilliseconds() const;
	float framesPerSecond() const;

	// Returns the average frame period, rounded to the nearest millisecond, as a QString
	QString framePeriodMillisecondsString() const;

	// Returns the average framerate, rounded to the nearest Hz, as a QString
	QString framesPerSecondString() const;	

private:

  QElapsedTimer m_clock;
	
	bool m_isFirstUpdate;
	qint64 m_lastUpdateTime;
	int m_nextSampleIndex;
	int m_nActualSamples;
	std::vector< qint64 > m_frameTimeSamples;

};
