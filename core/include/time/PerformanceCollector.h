#pragma once

#include <QString>
#include <QHash>
#include <QElapsedTimer>

class PerformanceCollector
{
public:

	PerformanceCollector();

	// register an event for performance collection
	// its event counter starts of as reset
	void registerEvent( QString name );

	// unregister an event for performance collection
	void unregisterEvent( QString name );

	// resets the statistics for an event
	void resetEvent( QString name );

	// call each time an event starts
	void beginEvent( QString name );

	// call each time an event ends
	void endEvent( QString name );

	// returns the average time spent on an event
	// over all beginEvent/endEvent pairs
	float averageTimeMilliseconds( QString name );

private:

  QElapsedTimer m_clock;
	QHash< QString, qint64 > m_eventStartTimes;	
	QHash< QString, qint64 > m_eventTotalElapsedTime;
	QHash< QString, int > m_eventCounts;

};
