#include "QKinectSpeechThread.h"

#include <chrono>

#include <QMetaType>
#include <QEventLoop>

#include <math/Arithmetic.h>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

QKinectSpeechThread::QKinectSpeechThread( std::shared_ptr< QKinect > pKinect, int pollingIntervalMS ) :

    m_pKinect( pKinect ),
    m_pollingIntervalMS( pollingIntervalMS )

{
    qRegisterMetaType< NUI_SKELETON_FRAME >( "NUI_SKELETON_FRAME" );
    qRegisterMetaType< Array2D< uint8x4 > >( "Array2D< uint8x4 >" );
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
    auto t0 = std::chrono::high_resolution_clock::now();

    QString phrase;
    float confidence;

    while( m_running )
    {
        bool detected = m_pKinect->pollSpeech( phrase, confidence );
        auto t1 = std::chrono::high_resolution_clock::now();

        if( detected )
        {
            emit speechRecognized( phrase, confidence );
        }

        int dtMS = std::chrono::duration_cast< std::chrono::milliseconds >(t1 - t0).count();
        t0 = t1;

        if( dtMS < m_pollingIntervalMS )
        {
            int sleepInterval = m_pollingIntervalMS - dtMS;
            msleep( sleepInterval );
        }

        eventLoop.processEvents();
    }
}
