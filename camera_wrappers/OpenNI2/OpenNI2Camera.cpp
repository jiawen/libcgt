#include "OpenNI2Camera.h"

#include "OpenNI2CameraImpl.h"

using libcgt::core::cameras::Intrinsics;
using libcgt::core::vecmath::EuclideanTransform;
using libcgt::camera_wrappers::openni2::OpenNI2CameraImpl;

namespace libcgt { namespace camera_wrappers { namespace openni2 {

// static
uint16_t OpenNI2Camera::minimumDepthMillimeters()
{
    return 500;
}

// static
uint16_t OpenNI2Camera::maximumDepthMillimeters()
{
    return 4000;
}

// static
Range1f OpenNI2Camera::depthRangeMeters()
{
    return Range1f::fromMinMax
    (
        minimumDepthMillimeters() * 0.001f,
        maximumDepthMillimeters() * 0.001f
    );
}

// static
EuclideanTransform OpenNI2Camera::colorFromDepthExtrinsicsMillimeters()
{
    // From {openni_camera,openni_camera_deperecated}/info/openni_params.yaml.
    return
    {
        Matrix3f
        (
             0.999979f,  0.006497f, -0.000801f,
            -0.006498f,  0.999978f, -0.001054f,
             0.000794f,  0.001059f,  0.999999f
        ),
        { -25.165f, 0.0047f, -0.4077f }
    };
}

// static
EuclideanTransform OpenNI2Camera::colorFromDepthExtrinsicsMeters()
{
    auto output = colorFromDepthExtrinsicsMillimeters();
    output.translation *= 0.001f;
    return output;
}

OpenNI2Camera::OpenNI2Camera( const std::vector< StreamConfig >& streamConfig,
    const char* uri ) :
    m_impl( new OpenNI2CameraImpl( streamConfig, uri ) )
{

}

OpenNI2Camera::~OpenNI2Camera()
{

}

bool OpenNI2Camera::isValid() const
{
    return m_impl->isValid();
}

Intrinsics OpenNI2Camera::colorIntrinsics() const
{
    return m_impl->colorIntrinsics();
}

Intrinsics OpenNI2Camera::depthIntrinsics() const
{
    return m_impl->depthIntrinsics();
}

StreamConfig OpenNI2Camera::colorConfig() const
{
    return m_impl->colorConfig();
}

StreamConfig OpenNI2Camera::depthConfig() const
{
    return m_impl->depthConfig();
}

StreamConfig OpenNI2Camera::infraredConfig() const
{
    return m_impl->infraredConfig();
}

void OpenNI2Camera::start()
{
    m_impl->start();
}

void OpenNI2Camera::stop()
{
    m_impl->stop();
}

bool OpenNI2Camera::getAutoExposureEnabled()
{
    return m_impl->getAutoExposureEnabled();
}

bool OpenNI2Camera::setAutoExposureEnabled( bool enabled )
{
    return m_impl->setAutoExposureEnabled( enabled );
}

int OpenNI2Camera::getExposure()
{
    return m_impl->getExposure();
}

int OpenNI2Camera::getGain()
{
    return m_impl->getGain();
}

bool OpenNI2Camera::setExposure( int exposure )
{
    return m_impl->setExposure( exposure );
}

bool OpenNI2Camera::setGain( int gain )
{
    return m_impl->setGain( gain );
}

bool OpenNI2Camera::getAutoWhiteBalanceEnabled()
{
    return m_impl->getAutoWhiteBalanceEnabled();
}

bool OpenNI2Camera::setAutoWhiteBalanceEnabled( bool enabled )
{
    return m_impl->setAutoExposureEnabled( enabled );
}

bool OpenNI2Camera::pollColor( OpenNI2Camera::Frame& frame, int timeoutMS )
{
    return m_impl->pollColor( frame, timeoutMS );
}

bool OpenNI2Camera::pollDepth( OpenNI2Camera::Frame& frame, int timeoutMS )
{
    return m_impl->pollDepth( frame, timeoutMS );
}

bool OpenNI2Camera::pollInfrared( OpenNI2Camera::Frame& frame, int timeoutMS )
{
    return m_impl->pollInfrared( frame, timeoutMS );
}

bool OpenNI2Camera::pollOne( Frame& frame, int timeoutMS )
{
    return m_impl->pollOne( frame, timeoutMS );
}

bool OpenNI2Camera::pollAll( Frame& frame, int timeoutMS )
{
    return m_impl->pollAll( frame, timeoutMS );
}

} } } // openni2, camera_wrappers, libcgt
