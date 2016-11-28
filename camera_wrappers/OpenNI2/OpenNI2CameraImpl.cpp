#include "OpenNI2CameraImpl.h"

#include <common/ArrayUtils.h>
#include <time/TimeUtils.h>
#include <cameras/CameraUtils.h>

using namespace openni;
using libcgt::core::arrayutils::copy;
using libcgt::core::arrayutils::componentView;
using libcgt::core::cameras::fovRadiansToFocalLengthPixels;
using libcgt::core::cameras::Intrinsics;
using libcgt::core::time::usToNS;
using libcgt::core::vecmath::EuclideanTransform;

namespace libcgt { namespace camera_wrappers { namespace openni2 {

bool OpenNI2CameraImpl::s_isOpenNIInitialized = false;

OpenNI2CameraImpl::OpenNI2CameraImpl(
    const std::vector< StreamConfig >& streamConfig, const char* uri )
{
    // TODO: do more validation: you can't have RGB and IR at the same
    // time, etc.
    // Then move it into a function and re-introduce const.
    for( int i = 0; i < static_cast< int >( streamConfig.size() ); ++i )
    {
        if( streamConfig[ i ].type == StreamType::COLOR )
        {
            m_colorConfig = streamConfig[ i ];
            break;
        }
    }

    for( int i = 0; i < static_cast< int >( streamConfig.size() ); ++i )
    {
        if( streamConfig[ i ].type == StreamType::DEPTH )
        {
            m_depthConfig = streamConfig[ i ];
            break;
        }
    }

    for( int i = 0; i < static_cast< int >( streamConfig.size() ); ++i )
    {
        if( streamConfig[ i ].type == StreamType::INFRARED )
        {
            m_infraredConfig = streamConfig[ i ];
            break;
        }
    }

    const bool enableSync = true;

    if( !OpenNI2CameraImpl::s_isOpenNIInitialized )
    {
        Status rc = OpenNI::initialize();
        if( rc == STATUS_OK )
        {
            OpenNI2CameraImpl::s_isOpenNIInitialized = true;
        }
    }

    if( OpenNI2CameraImpl::s_isOpenNIInitialized )
    {
        Status rc = m_device.open( uri );
        if( rc == STATUS_OK )
        {
            rc = m_device.setDepthColorSyncEnabled( enableSync );
            if( rc == STATUS_OK )
            {
                m_isValid = initializeStreams();
            }
        }
    }
}

// virtual
OpenNI2CameraImpl::~OpenNI2CameraImpl()
{
    stop();
    if( m_infraredStream.isValid() )
    {
        m_infraredStream.destroy();
    }
    if( m_depthStream.isValid() )
    {
        m_depthStream.destroy();
    }
    if( m_colorStream.isValid() )
    {
        m_colorStream.destroy();
    }
    m_device.close();

    // TODO: shutdown once?
    //OpenNI::shutdown();
    //s_isOpenNIInitialized = false;
}

bool OpenNI2CameraImpl::isValid() const
{
    return m_isValid;
}

Intrinsics OpenNI2CameraImpl::colorIntrinsics() const
{
    float fovX = m_colorStream.getHorizontalFieldOfView();
    float fovY = m_colorStream.getVerticalFieldOfView();

    return
    {
        {
            fovRadiansToFocalLengthPixels( fovX,
                static_cast< float >( m_colorConfig.resolution.x ) ),
            fovRadiansToFocalLengthPixels( fovY,
                static_cast< float >( m_colorConfig.resolution.y ) ),
        },
        {
            0.5f * m_colorConfig.resolution.x,
            0.5f * m_colorConfig.resolution.y
        }
    };
}

Intrinsics OpenNI2CameraImpl::depthIntrinsics() const
{
    float fovX = m_depthStream.getHorizontalFieldOfView();
    float fovY = m_depthStream.getVerticalFieldOfView();

    return
    {
        {
            fovRadiansToFocalLengthPixels( fovX,
                static_cast< float >( m_depthConfig.resolution.x ) ),
            fovRadiansToFocalLengthPixels( fovY,
                static_cast< float >( m_depthConfig.resolution.y ) ),
        },
        {
            0.5f * m_depthConfig.resolution.x,
            0.5f * m_depthConfig.resolution.y
        }
    };
}

StreamConfig OpenNI2CameraImpl::colorConfig() const
{
    return m_colorConfig;
}

StreamConfig OpenNI2CameraImpl::depthConfig() const
{
    return m_depthConfig;
}

StreamConfig OpenNI2CameraImpl::infraredConfig() const
{
    return m_infraredConfig;
}

bool OpenNI2CameraImpl::start()
{
    bool allSucceeded = true;

    if( m_colorStream.isValid() )
    {
        Status rc = m_colorStream.start();
        allSucceeded &= ( rc == STATUS_OK );
    }
    if( m_depthStream.isValid() )
    {
        Status rc = m_depthStream.start();
        allSucceeded &= ( rc == STATUS_OK );
    }
    if( m_infraredStream.isValid() )
    {
        Status rc = m_infraredStream.start();
        allSucceeded &= ( rc == STATUS_OK );
    }

    return allSucceeded;
}

void OpenNI2CameraImpl::stop()
{
    if( m_infraredStream.isValid() )
    {
        m_infraredStream.stop();
    }
    if( m_depthStream.isValid() )
    {
        m_depthStream.stop();
    }
    if( m_colorStream.isValid() )
    {
        m_colorStream.stop();
    }
}

int OpenNI2CameraImpl::getExposure()
{
    if( m_colorStream.isValid() )
    {
        return m_colorStream.getCameraSettings()->getExposure();
    }
    return 0;
}

int OpenNI2CameraImpl::getGain()
{
    if( m_colorStream.isValid() )
    {
        return m_colorStream.getCameraSettings()->getGain();
    }
    return 0;
}

bool OpenNI2CameraImpl::setExposure( int exposure )
{
    if( m_colorStream.isValid() )
    {
        Status rc = m_colorStream.getCameraSettings()->setExposure( exposure );
        return rc == STATUS_OK;
    }
    return false;
}

bool OpenNI2CameraImpl::setGain( int gain )
{
    if( m_colorStream.isValid() )
    {
        Status rc = m_colorStream.getCameraSettings()->setGain( gain );
        return rc == STATUS_OK;
    }
    return false;
}

bool OpenNI2CameraImpl::getAutoExposureEnabled()
{
    if( m_colorStream.isValid() )
    {
        return m_colorStream.getCameraSettings()->getAutoExposureEnabled();
    }
    return false;
}

bool OpenNI2CameraImpl::setAutoExposureEnabled( bool enabled )
{
    if( m_colorStream.isValid() )
    {
        Status rc = m_colorStream.getCameraSettings()->setAutoExposureEnabled(
            enabled );
        return rc == STATUS_OK;
    }
    return false;
}

bool OpenNI2CameraImpl::getAutoWhiteBalanceEnabled()
{
    if( m_colorStream.isValid() )
    {
        return m_colorStream.getCameraSettings()->getAutoWhiteBalanceEnabled();
    }
    return false;
}

bool OpenNI2CameraImpl::setAutoWhiteBalanceEnabled( bool enabled )
{
    if( m_colorStream.isValid() )
    {
        Status rc =
            m_colorStream.getCameraSettings()->setAutoWhiteBalanceEnabled(
                enabled );
        return rc == STATUS_OK;
    }
    return false;
}

bool OpenNI2CameraImpl::copyColor( OpenNI2Camera::FrameView& frame )
{
    frame.colorUpdated = false;

    VideoFrameRef src;
    Status rc = m_colorStream.readFrame( &src );
    if( rc == STATUS_OK )
    {
        Array2DReadView< uint8x3 > srcData( src.getData(),
            { src.getWidth(), src.getHeight() },
            { sizeof( uint8x3 ), src.getStrideInBytes() } );
        frame.colorTimestampNS =
            usToNS( static_cast< int64_t >( src.getTimestamp() ) );
        frame.colorFrameNumber = src.getFrameIndex();
        frame.colorUpdated = copy( srcData, frame.color );
        return frame.colorUpdated;
    }
    return false;
}

bool OpenNI2CameraImpl::copyDepth( OpenNI2Camera::FrameView& frame )
{
    frame.depthUpdated = false;

    VideoFrameRef src;
    Status rc = m_depthStream.readFrame( &src );
    if( rc == STATUS_OK )
    {
        Array2DReadView< uint16_t > srcData( src.getData(),
            { src.getWidth(), src.getHeight() },
            { sizeof( uint16_t ), src.getStrideInBytes() } );
        frame.depthTimestampNS =
            usToNS( static_cast< int64_t >( src.getTimestamp() ) );
        frame.depthFrameNumber = src.getFrameIndex();
        frame.depthUpdated = copy( srcData, frame.depth );
        return frame.depthUpdated;
    }
    return false;
}

bool OpenNI2CameraImpl::copyInfrared( OpenNI2Camera::FrameView& frame )
{
    frame.infraredUpdated = false;

    VideoFrameRef src;
    Status rc = m_infraredStream.readFrame( &src );
    if( rc == STATUS_OK )
    {
        Array2DReadView< uint16_t > srcData( src.getData(),
            { src.getWidth(), src.getHeight() },
            { sizeof( uint16_t ), src.getStrideInBytes() } );
        frame.infraredTimestampNS =
            usToNS( static_cast< int64_t >( src.getTimestamp() ) );
        frame.infraredFrameNumber = src.getFrameIndex();
        frame.infraredUpdated = copy( srcData, frame.infrared );
        return frame.infraredUpdated;
    }
    return false;
}

bool OpenNI2CameraImpl::pollColor( OpenNI2Camera::FrameView& frame, int timeoutMS )
{
    if( !m_colorStream.isValid() )
    {
        return false;
    }

    VideoStream* streams[] = { &m_colorStream };
    int readyIndex = -1; // Initialized to fail.
    Status rc =
        OpenNI::waitForAnyStream( streams, 1, &readyIndex, timeoutMS );
    if( rc == STATUS_OK && readyIndex == 0 )
    {
        return copyColor( frame );
    }
    return false;
}

bool OpenNI2CameraImpl::pollDepth( OpenNI2Camera::FrameView& frame, int timeoutMS )
{
    if( !m_depthStream.isValid() )
    {
        return false;
    }

    VideoStream* streams[] = { &m_depthStream };
    int readyIndex = -1; // Initialized to fail.
    Status rc =
        OpenNI::waitForAnyStream( streams, 1, &readyIndex, timeoutMS );
    if( rc == STATUS_OK && readyIndex == 0 )
    {
        return copyDepth( frame );
    }
    return false;
}

bool OpenNI2CameraImpl::pollInfrared( OpenNI2Camera::FrameView& frame, int timeoutMS )
{
    if( !m_infraredStream.isValid() )
    {
        return false;
    }

    VideoStream* streams[] = { &m_infraredStream };
    int readyIndex = -1; // Initialized to fail.
    Status rc =
        OpenNI::waitForAnyStream( streams, 1, &readyIndex, timeoutMS );
    if( rc == STATUS_OK && readyIndex == 0 )
    {
        return copyInfrared( frame );
    }
    return false;
}

bool OpenNI2CameraImpl::pollOne( OpenNI2Camera::FrameView& frame, int timeoutMS )
{
    frame.colorUpdated = false;
    frame.depthUpdated = false;
    frame.infraredUpdated = false;

    VideoStream* streams[] =
    {
        &m_colorStream,
        &m_depthStream,
        &m_infraredStream
    };

    int readyIndex = -1; // Initialized to fail.
    Status rc =
        OpenNI::waitForAnyStream( streams, 3, &readyIndex, timeoutMS );
    if( rc == STATUS_OK )
    {
        if( readyIndex == 0 ) // color
        {
            return copyColor( frame );
        }
        else if( readyIndex == 1 ) // depth
        {
            return copyDepth( frame );
        }
        else if( readyIndex == 2 ) // infrared
        {
            return copyInfrared( frame );
        }
    }
    return false;
}

bool OpenNI2CameraImpl::pollAll( OpenNI2Camera::FrameView& frame, int timeoutMS )
{
    frame.colorUpdated = false;
    frame.depthUpdated = false;
    frame.infraredUpdated = false;

    bool allSucceeded = true;
    if( m_colorStream.isValid() )
    {
        allSucceeded &= pollColor( frame, timeoutMS );
    }
    if( m_depthStream.isValid() )
    {
        allSucceeded &= pollDepth( frame, timeoutMS );
    }
    if( m_infraredStream.isValid() )
    {
        allSucceeded &= pollInfrared( frame, timeoutMS );
    }
    return allSucceeded;
}

// Returns static_cast< openni::PixelFormat >( 0 ) if no match found.
openni::PixelFormat toOpenNI( PixelFormat format )
{
    switch( format )
    {
    case PixelFormat::DEPTH_MM_U16:
        return PIXEL_FORMAT_DEPTH_1_MM;
    case PixelFormat::RGB_U888:
        return PIXEL_FORMAT_RGB888;
    case PixelFormat::GRAY_U8:
        return PIXEL_FORMAT_GRAY8;
    case PixelFormat::GRAY_U16:
        return PIXEL_FORMAT_GRAY16;
    default:
        return static_cast< openni::PixelFormat >( 0 );
    }
}

int findMatchingVideoModeIndex( StreamConfig config,
    const Array< VideoMode >& modes )
{
    int index = -1;
    for( int i = 0; i < modes.getSize(); ++i )
    {
        const auto& mode = modes[ i ];
        int width = mode.getResolutionX();
        int height = mode.getResolutionY();
        int fps = mode.getFps();
        auto pixelFormat = mode.getPixelFormat();
        if( width == config.resolution.x &&
            height == config.resolution.y &&
            fps == config.fps &&
            pixelFormat == toOpenNI( config.pixelFormat ) )
        {
            index = i;
            break;
        }
    }
    return index;
}

bool OpenNI2CameraImpl::initializeStreams()
{
    // request color, has color, resolution, etc
    const SensorInfo* colorInfo =
        m_device.getSensorInfo( SENSOR_COLOR );

    const SensorInfo* depthInfo =
        m_device.getSensorInfo( SENSOR_DEPTH );

    const SensorInfo* infraredInfo =
        m_device.getSensorInfo( SENSOR_IR );

    const auto& colorModes = colorInfo->getSupportedVideoModes();
    const auto& depthModes = depthInfo->getSupportedVideoModes();
    const auto& infraredModes = infraredInfo->getSupportedVideoModes();

    int colorModeIndex =
        findMatchingVideoModeIndex( m_colorConfig, colorModes );
    int depthModeIndex =
        findMatchingVideoModeIndex( m_depthConfig, depthModes );
    int infraredModeIndex =
        findMatchingVideoModeIndex( m_infraredConfig, infraredModes );

    if( colorModeIndex != -1 )
    {
        m_colorStream.create( m_device, SENSOR_COLOR );
        m_colorStream.setVideoMode( colorModes[ colorModeIndex ] );
        m_colorStream.setMirroringEnabled( m_colorConfig.mirror );
    }

    if( depthModeIndex != -1 )
    {
        m_depthStream.create( m_device, SENSOR_DEPTH );
        m_depthStream.setVideoMode( depthModes[ depthModeIndex ] );
        m_depthStream.setMirroringEnabled( m_depthConfig.mirror );
    }

    if( infraredModeIndex != -1 )
    {
        m_infraredStream.create( m_device, SENSOR_IR );
        m_infraredStream.setVideoMode( infraredModes[ infraredModeIndex ] );
        m_infraredStream.setMirroringEnabled( m_infraredConfig.mirror );
    }

    return( colorModeIndex != -1 ||
        depthModeIndex != -1 ||
        infraredModeIndex != -1 );
}

} } } // openni2, camera_wrappers, libcgt
