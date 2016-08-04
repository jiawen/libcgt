#include "QKinectThread.h"

#include <chrono>

#include <QMetaType>
#include <QEventLoop>

#include <math/Arithmetic.h>
#include <vector>

QKinectThread::QKinectThread( std::shared_ptr< QKinect > pKinect, int pollingIntervalMS ) :

    m_pKinect( pKinect ),
    m_pollingIntervalMS( pollingIntervalMS )

{
    qRegisterMetaType< NUI_SKELETON_FRAME >( "NUI_SKELETON_FRAME" );
    qRegisterMetaType< Array2D< uint8x3 > >( "Array2D< uint8x3 >" );
    qRegisterMetaType< Array2D< uint16_t > >( "Array2D< uint16_t >" );
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

// virtual
void QKinectThread::run()
{
    m_running = true;
    QEventLoop eventLoop;
    auto tStart = std::chrono::high_resolution_clock::now();
    auto t0 = tStart;

    NUI_SKELETON_FRAME skeletonFrame;
    // TODO: pass in resolutions to QKinect on initialization
    // query it to get the sizes
    Array2D< uint8x4 > bgra( { 640, 480 } );
    Array2D< uint16_t > depth( { 640, 480 } );
    Array2D< uint16_t > playerIndex( { 640, 480 } );

    int64_t bgraTimestamp;
    int bgraFrameNumber;
    int64_t depthTimestamp;
    int depthFrameNumber;
    bool depthCapturedWithNearMode;

    while( m_running )
    {
        QKinect::Event e = m_pKinect->poll( skeletonFrame,
            bgra, bgraTimestamp, bgraFrameNumber,
            depth, playerIndex,
            depthTimestamp, depthFrameNumber, depthCapturedWithNearMode,
            m_pollingIntervalMS );
        auto t1 = std::chrono::high_resolution_clock::now();

        switch( e )
        {
        case QKinect::Event::SKELETON:
            {
                emit skeletonFrameReady( skeletonFrame );
                break;
            }
        case QKinect::Event::COLOR:
            {
                emit colorFrameReady( bgra );
                break;
            }
        case QKinect::Event::DEPTH:
            {
                emit depthFrameReady( depth );
                break;
            }
        }

        int dtMS = std::chrono::duration_cast< std::chrono::milliseconds >( t1 - t0 ).count();
        t0 = t1;

        // printf( "dt = %f ms\n", dt );

        if( dtMS < m_pollingIntervalMS )
        {
            int sleepInterval = m_pollingIntervalMS - dtMS;
            //printf( "sleeping for %d milliseconds\n", sleepInterval );
            msleep( sleepInterval );
        }

        //int64 t2 = clock.getCounterValue();
        //float loopTime = clock.convertIntervalToMillis( t2 - t1 );
        //printf( "loopTime = %f ms\n", loopTime );
        // msleep( m_pollingIntervalMS );

        eventLoop.processEvents();
    }
}
