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
	int64 tStart = clock.getCounterValue();
	int64 t0 = tStart;

	NUI_SKELETON_FRAME skeletonFrame;
	Image4ub rgba( 640, 480 );
	Array2D< ushort > depth( 640, 480 );

	while( m_running )
	{
		QKinect::QKinectEvent e = m_pKinect->poll( skeletonFrame, rgba, depth, m_pollingIntervalMS );
		int64 t1 = clock.getCounterValue();		

		switch( e )
		{
		case QKinect::QKinect_Event_Skeleton:
			{
				emit skeletonFrameReady( skeletonFrame ); 
				break;
			}
		case QKinect::QKinect_Event_RGB:
			{
				emit colorFrameReady( rgba );
				break;
			}
		case QKinect::QKinect_Event_Depth:
			{
				emit depthFrameReady( depth );
				break;
			}
		}

#if 0
		float dt = clock.convertIntervalToMillis( t1 - t0 );
		t0 = t1;
		
		printf( "dt = %f ms\n", dt );

		if( dt < m_pollingIntervalMS )
		{
			int sleepInterval = Arithmetic::roundToInt( m_pollingIntervalMS - dt );			
			printf( "sleeping for %d milliseconds\n", sleepInterval );
			msleep( sleepInterval );
		}

		int64 t2 = clock.getCounterValue();
		float loopTime = clock.convertIntervalToMillis( t2 - t1 );
		printf( "loopTime = %f ms\n", loopTime );
#endif

		msleep( m_pollingIntervalMS );

		eventLoop.processEvents();
	}
}
