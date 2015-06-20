#include "QKinectSpeechThread.h"

#include <QMetaType>
#include <QEventLoop>

#include <time/Clock.h>
#include <math/Arithmetic.h>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

QKinectSpeechThread::QKinectSpeechThread( std::shared_ptr< QKinect > pKinect, int pollingIntervalMS ) :

    m_pKinect( pKinect ),
    m_pollingIntervalMS( pollingIntervalMS )

{
    qRegisterMetaType< NUI_SKELETON_FRAME >( "NUI_SKELETON_FRAME" );
    qRegisterMetaType< Image4ub >( "Image4ub" );
}

int QKinectSpeechThread::pollingIntervalMS() const
{
    return m_pollingIntervalMS;
}

void QKinectSpeechThread::setPollingIntervalMS( int pollingIntervalMS )
{
    m_pollingIntervalMS = pollingIntervalMS;
}

void QKinectSpeechThread::stop()
{
    m_running = false;
}

//////////////////////////////////////////////////////////////////////////
// Protected
//////////////////////////////////////////////////////////////////////////

// virtual
void QKinectSpeechThread::run()
{
    m_running = true;
    QEventLoop eventLoop;
    Clock clock;
    int64 t0 = clock.getCounterValue();

    QString phrase;
    float confidence;

    while( m_running )
    {
        bool detected = m_pKinect->pollSpeech( phrase, confidence );
        int64 t1 = clock.getCounterValue();

        if( detected )
        {
            emit speechRecognized( phrase, confidence );
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
