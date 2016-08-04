#pragma once

#include <memory>

#include <QThread>

#include "QKinect.h"

class QKinectSpeechThread : public QThread
{
    Q_OBJECT

public:

    QKinectSpeechThread( std::shared_ptr< QKinect > pKinect, int pollingIntervalMS = 30 );

    int pollingIntervalMS() const;
    void setPollingIntervalMS( int pollingIntervalMS );

    void stop();

signals:

    void speechRecognized( QString phrase, float confidence );

protected:

    virtual void run();

private:

    bool m_running;
    int m_pollingIntervalMS;
    std::shared_ptr< QKinect > m_pKinect;
};
