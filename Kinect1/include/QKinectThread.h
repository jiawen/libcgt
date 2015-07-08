#pragma once

#include <memory>

#include <Windows.h>
#include <NuiApi.h>

#include <QThread>

#include <common/BasicTypes.h>
#include <common/Array2D.h>

#include "QKinect.h"

// TODO: Get rid of QThread, use std::thread.
// TODO: Get rid of std::shared_ptr: use std::unique_ptr.
class QKinectThread : public QThread
{
    Q_OBJECT

public:

    QKinectThread( std::shared_ptr< QKinect > pKinect, int pollingIntervalMS = 30 );

    int pollingIntervalMS() const;
    void setPollingIntervalMS( int pollingIntervalMS );

    void stop();

    std::shared_ptr< QKinect > kinect();

signals:

    void skeletonFrameReady( const NUI_SKELETON_FRAME& skeletonFrame );
    void colorFrameReady( const Array2D< uint8x4 >& bgra );
    void depthFrameReady( const Array2D< uint16_t >& depth );

protected:

    virtual void run();

private:

    bool m_running;
    int m_pollingIntervalMS;
    std::shared_ptr< QKinect > m_pKinect;
};
