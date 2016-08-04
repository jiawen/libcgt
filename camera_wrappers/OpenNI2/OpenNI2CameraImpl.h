#pragma once

#include <OpenNI.h>

#include "OpenNI2Camera.h"

namespace libcgt { namespace camera_wrappers { namespace openni2 {

class OpenNI2CameraImpl
{
public:

    using Intrinsics = libcgt::core::cameras::Intrinsics;
    using EuclideanTransform = libcgt::core::vecmath::EuclideanTransform;

    OpenNI2CameraImpl(
        StreamConfig colorConfig =
            StreamConfig{ { 640, 480 }, 30, PixelFormat::RGB_U888 },
        StreamConfig depthConfig =
            StreamConfig{ { 640, 480 }, 30, PixelFormat::DEPTH_MM_U16 },
        StreamConfig infraredConfig = StreamConfig(),
        const char* uri = openni::ANY_DEVICE );
    virtual ~OpenNI2CameraImpl();
    // TODO(VS2015): move constructor = default

    bool isValid() const;

    Intrinsics colorIntrinsics() const;
    Intrinsics depthIntrinsics() const;

    StreamConfig colorConfig() const;
    StreamConfig depthConfig() const;
    StreamConfig infraredConfig() const;

    void start();
    void stop();

    bool pollOne( OpenNI2Camera::Frame& frame, int timeoutMS = 0 );

    // Poll all registered streams.
    // Returns true if all succeeded.
    bool pollAll( OpenNI2Camera::Frame& frame, int timeoutMS = 0 );

private:

    bool initializeStreams();

    openni::Device m_device;

    const StreamConfig m_colorConfig;
    const StreamConfig m_depthConfig;
    const StreamConfig m_infraredConfig;

    openni::VideoStream m_colorStream;
    openni::VideoStream m_depthStream;
    openni::VideoStream m_infraredStream;

    bool m_isValid = false;
    static bool s_isOpenNIInitialized;
};

} } } // openni2, camera_wrappers, libcgt
