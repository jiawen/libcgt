#include "OpenNI2Camera.h"

#include "OpenNI2CameraImpl.h"

using libcgt::core::cameras::Intrinsics;
using libcgt::camera_wrappers::openni2::OpenNI2CameraImpl;

namespace libcgt { namespace camera_wrappers { namespace openni2 {

OpenNI2Camera::OpenNI2Camera( StreamConfig colorConfig,
    StreamConfig depthConfig, StreamConfig infraredConfig, const char* uri ) :
    m_impl( new OpenNI2CameraImpl(
        colorConfig, depthConfig, infraredConfig, uri ) )
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

bool OpenNI2Camera::pollOne( Frame& frame, int timeoutMS )
{
    return m_impl->pollOne( frame, timeoutMS );
}

bool OpenNI2Camera::pollAll( Frame& frame, int timeoutMS )
{
    return m_impl->pollAll( frame, timeoutMS );
}

} } } // openni2, camera_wrappers, libcgt
