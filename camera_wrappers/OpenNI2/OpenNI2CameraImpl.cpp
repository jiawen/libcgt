#include "OpenNI2CameraImpl.h"

#include <common/ArrayUtils.h>
#include <cameras/CameraUtils.h>

using namespace openni;
using libcgt::core::arrayutils::copy;
using libcgt::core::arrayutils::componentView;
using libcgt::core::cameras::fovRadiansToFocalLengthPixels;
using libcgt::core::cameras::Intrinsics;
using libcgt::core::vecmath::EuclideanTransform;

namespace libcgt { namespace camera_wrappers { namespace openni2 {

bool OpenNI2CameraImpl::s_isOpenNIInitialized = false;

OpenNI2CameraImpl::OpenNI2CameraImpl( StreamConfig colorConfig,
    StreamConfig depthConfig, StreamConfig infraredConfig, const char* uri ) :
    m_colorConfig( colorConfig ),
    m_depthConfig( depthConfig ),
    m_infraredConfig( infraredConfig )
{
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
            m_isValid = initializeStreams();
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

void OpenNI2CameraImpl::start()
{
    if( m_colorStream.isValid() )
    {
        m_colorStream.start();
    }
    if( m_depthStream.isValid() )
    {
        m_depthStream.start();
    }
    if( m_infraredStream.isValid() )
    {
        m_infraredStream.start();
    }
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

bool OpenNI2CameraImpl::pollOne( OpenNI2Camera::Frame& frame, int timeoutMS )
{
    frame.colorUpdated = false;
    frame.depthUpdated = false;

    VideoStream** streams = new VideoStream*[ 2 ];
    streams[ 0 ] = &m_colorStream;
    streams[ 1 ] = &m_depthStream;
    const int nStreams = 2;

    //VideoStream* streams[] = { &m_colorStream, &m_depthStream, &m_infraredStream };
    //const int nStreams = 3;

    int readyIndex = -1; // Initialized to fail.
    Status rc =
        OpenNI::waitForAnyStream( streams, nStreams, &readyIndex, timeoutMS );
    if( rc == STATUS_OK )
    {
        if( readyIndex == 0 ) // color
        {
            VideoFrameRef src;
            // TODO(jiawen): this returns a status also
            m_colorStream.readFrame( &src );

            Array2DView< const uint8x3 > srcData( src.getData(),
                { src.getWidth(), src.getHeight() },
                { sizeof( uint8x3 ), src.getStrideInBytes() } );
            frame.colorUpdated = copy( srcData, frame.rgb );
            frame.colorTimestamp =
                static_cast< int64_t >( src.getTimestamp() );
            frame.colorFrameNumber = src.getFrameIndex();

            return true;
        }
        else if( readyIndex == 1 ) // depth
        {
            VideoFrameRef src;
            // TODO(jiawen): this returns a status also
            m_depthStream.readFrame( &src );

            Array2DView< const uint16_t > srcData( src.getData(),
                { src.getWidth(), src.getHeight() },
                { sizeof( uint16_t ), src.getStrideInBytes() } );
            frame.depthUpdated = copy( srcData, frame.depth );
            frame.depthTimestamp =
                static_cast< int64_t >( src.getTimestamp() );
            frame.depthFrameNumber = src.getFrameIndex();

            return true;
        }
        else if( readyIndex == 2 ) // infrared
        {
            VideoFrameRef src;
            // TODO(jiawen): this returns a status also
            m_infraredStream.readFrame( &src );

            Array2DView< const uint16_t > srcData( src.getData(),
                { src.getWidth(), src.getHeight() },
                { sizeof( uint16_t ), src.getStrideInBytes() } );
            frame.colorUpdated = copy( srcData, frame.infrared );
            frame.colorTimestamp =
                static_cast< int64_t >( src.getTimestamp() );
            frame.colorFrameNumber = src.getFrameIndex();

            return true;
        }
    }

    delete[] streams;

    return false;
}

bool OpenNI2CameraImpl::pollAll( OpenNI2Camera::Frame& frame, int timeoutMS )
{
    // TODO: put registered frames into an std::vector and iterate on pollOne()

    frame.colorUpdated = false;
    frame.depthUpdated = false;

    int readyIndex = -1; // Initialized to fail.
    VideoStream* streams[ 1 ];

    Status rc;

    bool allSucceeded = true;

    if( m_colorStream.isValid() )
    {
        streams[ 0 ] = &m_colorStream;
        rc = OpenNI::waitForAnyStream( streams, 1, &readyIndex, timeoutMS );
        if( rc == STATUS_OK )
        {
            VideoFrameRef src;
            // TODO(jiawen): this returns a status also
            m_colorStream.readFrame( &src );

            Array2DView< const uint8x3 > srcData( src.getData(),
            { src.getWidth(), src.getHeight() },
            { sizeof( uint8x3 ), src.getStrideInBytes() } );
            frame.colorUpdated = copy( srcData, frame.rgb );
            frame.colorTimestamp =
                static_cast< int64_t >( src.getTimestamp() );
            frame.colorFrameNumber = src.getFrameIndex();
        }
        else
        {
            allSucceeded = false;
        }
    }

    if( m_depthStream.isValid() )
    {
        streams[ 0 ] = &m_depthStream;
        rc = OpenNI::waitForAnyStream( streams, 1, &readyIndex, timeoutMS );
        if( rc == STATUS_OK )
        {
            VideoFrameRef src;
            // TODO(jiawen): this returns a status also
            m_depthStream.readFrame( &src );

            Array2DView< const uint16_t > srcData( src.getData(),
            { src.getWidth(), src.getHeight() },
            { sizeof( uint16_t ), src.getStrideInBytes() } );
            frame.depthUpdated = copy( srcData, frame.depth );
            frame.depthTimestamp =
                static_cast< int64_t >( src.getTimestamp() );
            frame.depthFrameNumber = src.getFrameIndex();
        }
        else
        {
            allSucceeded = false;
        }
    }

    if( m_infraredStream.isValid() )
    {
        streams[ 0 ] = &m_infraredStream;
        rc = OpenNI::waitForAnyStream( streams, 1, &readyIndex, timeoutMS );
        if( rc == STATUS_OK )
        {
            // TODO(jiawen): refactor the reading into a function
            VideoFrameRef src;
            // TODO(jiawen): this returns a status also
            m_colorStream.readFrame( &src );

            Array2DView< const uint8x3 > srcData( src.getData(),
            { src.getWidth(), src.getHeight() },
            { sizeof( uint8x3 ), src.getStrideInBytes() } );
            frame.colorUpdated = copy( srcData, frame.rgb );
            frame.colorTimestamp =
                static_cast< int64_t >( src.getTimestamp() );
            frame.colorFrameNumber = src.getFrameIndex();
        }
        else
        {
            allSucceeded = false;
        }
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

#if 1
    if( depthModeIndex != -1 )
    {
        Status rc = m_depthStream.create( m_device, SENSOR_DEPTH );
        rc = m_depthStream.setVideoMode( depthModes[ depthModeIndex ] );
    }

    if( colorModeIndex != -1 )
    {
        Status rc = m_colorStream.create( m_device, SENSOR_COLOR );
        rc = m_colorStream.setVideoMode( colorModes[ colorModeIndex ] );
    }

    //Status rc = m_depthStream.start();
    //rc = m_colorStream.start();


#else
    if( depthModeIndex != -1 )
    {
        m_depthStream.create( m_device, SENSOR_DEPTH );
        m_depthStream.start();
    }

    if( colorModeIndex != -1 )
    {
        m_colorStream.create( m_device, SENSOR_COLOR );
        m_colorStream.start();
    }

    //m_colorStream.setVideoMode( colorModes[ colorModeIndex ] );
    //m_depthStream.setVideoMode( depthModes[ depthModeIndex ] );
#endif

    if( infraredModeIndex != -1 )
    {
        m_infraredStream.create( m_device, SENSOR_IR );
        m_infraredStream.setVideoMode( infraredModes[ infraredModeIndex ] );
        m_infraredStream.setMirroringEnabled( true );
    }

    return( colorModeIndex != -1 ||
        depthModeIndex != -1 ||
        infraredModeIndex != -1 );
}

} } } // openni2, camera_wrappers, libcgt
