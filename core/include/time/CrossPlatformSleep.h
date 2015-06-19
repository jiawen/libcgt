#pragma once

#include <QThread>

class CrossPlatformSleep : QThread
{
public:

	static void msleep( unsigned long milliseconds );
	static void usleep( unsigned long microseconds );

	// calls usleep after converting to microseconds
	static void sleep( float seconds );

	static void sleep( unsigned long seconds );

	// yields this thread by calling usleep( 0 );
	static void yieldThread();
};
