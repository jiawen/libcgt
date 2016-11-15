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
Intrinsics OpenNI2Camera::defaultColorIntrinsics()
{
    return
    {
        { 529.21508098293293f, 525.56393630057437f },
        { 328.94272028759258f, 267.48068171871557f }
#if 0
        k1_rgb 2.6451622333009589e-01
        k2_rgb - 8.3990749424620825e-01
        p1_rgb - 1.9922302173693159e-03
        p2_rgb 1.4371995932897616e-03
        k3_rgb 9.1192465078713847e-01
#endif
    };
}

// static
Intrinsics OpenNI2Camera::defaultDepthIntrinsics()
{
    return
    {
        { 594.21434211923247f, 591.04053696870778f },
        { 339.30780975300314f, 242.73913761751615f }
#if 0
        k1_d - 2.6386489753128833e-01
        k2_d 9.9966832163729757e-01
        p1_d - 7.6275862143610667e-04
        p2_d 5.0350940090814270e-03
        k3_d - 1.3053628089976321e+00
#endif
    };
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

bool OpenNI2Camera::pollColor( FrameView& frame, int timeoutMS )
{
    return m_impl->pollColor( frame, timeoutMS );
}

bool OpenNI2Camera::pollDepth( FrameView& frame, int timeoutMS )
{
    return m_impl->pollDepth( frame, timeoutMS );
}

bool OpenNI2Camera::pollInfrared( FrameView& frame, int timeoutMS )
{
    return m_impl->pollInfrared( frame, timeoutMS );
}

bool OpenNI2Camera::pollOne( FrameView& frame, int timeoutMS )
{
    return m_impl->pollOne( frame, timeoutMS );
}

bool OpenNI2Camera::pollAll( FrameView& frame, int timeoutMS )
{
    return m_impl->pollAll( frame, timeoutMS );
}

} } } // openni2, camera_wrappers, libcgt
