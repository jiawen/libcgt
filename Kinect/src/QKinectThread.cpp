#include "QKinectThread.h"

#include <QMetaType>
#include <QEventLoop>

#include <time/Clock.h>
#include <math/Arithmetic.h>
#include <vector>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

QKinectThread::QKinectThread( std::shared_ptr< QKinect > pKinect, int pollingIntervalMS ) :

	m_pKinect( pKinect ),
	m_pollingIntervalMS( pollingIntervalMS )

{
	qRegisterMetaType< NUI_SKELETON_FRAME >( "NUI_SKELETON_FRAME" );
	qRegisterMetaType< Image4ub >( "Image4ub" );
	qRegisterMetaType< Array2D< ushort > >( "Array2D< ushort >" );
}

int QKinectThread::pollingIntervalMS() const
{
	return m_pollingIntervalMS;
}

void QKinectThread::setPollingIntervalMS( int pollingIntervalMS )
{
	m_pollingIntervalMS = pollingIntervalMS;
}

void QKinectThread::stop()
{
	m_running = false;
}

//////////////////////////////////////////////////////////////////////////
// Protected
//////////////////////////////////////////////////////////////////////////

// virtual
void QKinectThread::run()
{
	m_running = true;
	QEventLoop eventLoop;
	Clock clock;
	int64 t0 = clock.getCounterValue();

	NUI_SKELETON_FRAME skeletonFrame;
	Image4ub rgbaFrame( 640, 480 );
	Array2D< ushort > depthFrame( 640, 480 );

	std::vector< bool > successMask;

	while( m_running )
	{
		successMask = m_pKinect->poll( skeletonFrame, rgbaFrame, depthFrame, m_pollingIntervalMS );
		int64 t1 = clock.getCounterValue();		

		if( successMask[ 2 ] )
		{
			emit skeletonFrameReady( skeletonFrame ); 
		}
		else if( successMask[ 1 ] )
		{
			emit colorFrameReady( rgbaFrame );
		}
		else if( successMask[ 0 ] )
		{
			emit depthFrameReady( depthFrame );
		}
		float dt = clock.convertIntervalToMillis( t1 - t0 );
		t0 = t1;

		if( dt < m_pollingIntervalMS )
		{
			int sleepInterval = Arithmetic::roundToInt( m_pollingIntervalMS - dt );
			msleep( sleepInterval );
		}

		eventLoop.processEvents();
	}
}
