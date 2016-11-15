#pragma once

#include <OpenNI.h>

#include "OpenNI2Camera.h"

namespace libcgt { namespace camera_wrappers { namespace openni2 {

class OpenNI2CameraImpl
{
public:

    using Intrinsics = libcgt::core::cameras::Intrinsics;
    using EuclideanTransform = libcgt::core::vecmath::EuclideanTransform;

    OpenNI2CameraImpl( const std::vector< StreamConfig >& streamConfig,
        const char* uri );
    virtual ~OpenNI2CameraImpl();

    bool isValid() const;

    Intrinsics colorIntrinsics() const;
    Intrinsics depthIntrinsics() const;

    StreamConfig colorConfig() const;
    StreamConfig depthConfig() const;
    StreamConfig infraredConfig() const;

    bool start();
    void stop();

    bool getAutoExposureEnabled();
    bool setAutoExposureEnabled( bool enabled );

    int getExposure();
    int getGain();
    bool setExposure( int exposure );
    bool setGain( int gain );

    bool getAutoWhiteBalanceEnabled();
    bool setAutoWhiteBalanceEnabled( bool enabled );

    bool pollColor( OpenNI2Camera::FrameView& frame, int timeoutMS = 0 );

    bool pollDepth( OpenNI2Camera::FrameView& frame, int timeoutMS = 0 );

    bool pollInfrared( OpenNI2Camera::FrameView& frame, int timeoutMS = 0 );

    bool pollOne( OpenNI2Camera::FrameView& frame, int timeoutMS = 0 );

    bool pollAll( OpenNI2Camera::FrameView& frame, int timeoutMS = 0 );

private:

    bool initializeStreams();

    bool copyColor( OpenNI2Camera::FrameView& frame );
    bool copyDepth( OpenNI2Camera::FrameView& frame );
    bool copyInfrared( OpenNI2Camera::FrameView& frame );

    openni::Device m_device;

    StreamConfig m_colorConfig;
    StreamConfig m_depthConfig;
    StreamConfig m_infraredConfig;

    openni::VideoStream m_colorStream;
    openni::VideoStream m_depthStream;
    openni::VideoStream m_infraredStream;

    bool m_isValid = false;
    static bool s_isOpenNIInitialized;
};

} } } // openni2, camera_wrappers, libcgt
