#include "libcgt/camera_wrappers/Kinect1x/KinectCamera.h"

#if KINECT1X_ENABLE_SPEECH
// For string IO and manipulation.
#include <strsafe.h>
#include <conio.h>
#endif

#include <Windows.h>
#include <NuiApi.h>

#include <common/ArrayUtils.h>
#include <imageproc/Swizzle.h>

#include "libcgt/camera_wrappers/kinect1x/KinectUtils.h"
#include "libcgt/camera_wrappers/kinect1x/KinectCameraImpl.h"

using libcgt::core::arrayutils::componentView;
using libcgt::core::arrayutils::copy;
using libcgt::core::cameras::Intrinsics;
using libcgt::core::vecmath::EuclideanTransform;

namespace libcgt { namespace camera_wrappers { namespace kinect1x {

// static
int KinectCamera::numDevices()
{
    int nDevices;
    NuiGetSensorCount( &nDevices );

    return nDevices;
}

// static
uint16_t KinectCamera::minimumDepthMillimeters()
{
    return NUI_IMAGE_DEPTH_MINIMUM >> NUI_IMAGE_PLAYER_INDEX_SHIFT;
}

// static
uint16_t KinectCamera::maximumDepthMillimeters()
{
    return NUI_IMAGE_DEPTH_MAXIMUM >> NUI_IMAGE_PLAYER_INDEX_SHIFT;
}

// static
Range1f KinectCamera::depthRangeMeters()
{
    return Range1f::fromMinMax
    (
        minimumDepthMillimeters() * 0.001f,
        maximumDepthMillimeters() * 0.001f
    );
}

// static
uint16_t KinectCamera::nearModeMinimumDepthMillimeters()
{
    return NUI_IMAGE_DEPTH_MINIMUM_NEAR_MODE >> NUI_IMAGE_PLAYER_INDEX_SHIFT;
}

// static
uint16_t KinectCamera::nearModeMaximumDepthMillimeters()
{
    return NUI_IMAGE_DEPTH_MAXIMUM_NEAR_MODE >> NUI_IMAGE_PLAYER_INDEX_SHIFT;
}

// static
Range1f KinectCamera::nearModeDepthRangeMeters()
{
    return Range1f::fromMinMax
    (
        nearModeMinimumDepthMillimeters() * 0.001f,
        nearModeMaximumDepthMillimeters() * 0.001f
    );
}

// static
Intrinsics KinectCamera::colorIntrinsics( const Vector2i& resolution )
{
    // NUI_CAMERA_COLOR_NOMINAL_FOCAL_LENGTH_IN_PIXELS is #defined for a
    // 640x480 image.
    Vector2f fl{ NUI_CAMERA_COLOR_NOMINAL_FOCAL_LENGTH_IN_PIXELS };
    Vector2f pp{ 320, 240 };

    if( resolution == Vector2i{ 640, 480 } )
    {
        // Do nothing.
    }
    else if( resolution == Vector2i{ 1280, 960 } )
    {
        fl *= 2;
        pp *= 2;
    }
    else
    {
        fl = Vector2f{ 0, 0 };
        pp = Vector2f{ 0, 0 };
    }

    return Intrinsics{ fl, pp };
}

// static
Intrinsics KinectCamera::depthIntrinsics( const Vector2i& resolution )
{
    // NUI_CAMERA_COLOR_NOMINAL_FOCAL_LENGTH_IN_PIXELS is #defined for a
    // 320x240 image.
    Vector2f fl{ NUI_CAMERA_DEPTH_NOMINAL_FOCAL_LENGTH_IN_PIXELS };
    Vector2f pp{ 160, 120 };

    if( resolution == Vector2i{ 80, 60 } )
    {
        fl *= 0.25f;
        pp *= 0.25f;
    }
    else if( resolution == Vector2i{ 320, 240 } )
    {
        // Do nothing.
    }
    else if( resolution == Vector2i{ 640, 480 } )
    {
        fl *= 2;
        pp *= 2;
    }
    else
    {
        fl = Vector2f{ 0, 0 };
        pp = Vector2f{ 0, 0 };
    }

    return Intrinsics{ fl, pp };
}

// static
EuclideanTransform KinectCamera::colorFromDepthExtrinsicsMillimeters()
{
    return
    {
        Matrix3f::identity(),
        { -25.4f, -0.13f, -2.18f }
    };
}

// static
EuclideanTransform KinectCamera::colorFromDepthExtrinsicsMeters()
{
    auto output = colorFromDepthExtrinsicsMillimeters();
    output.translation *= 0.001f;
    return output;
}

#if 0
// static
const std::vector< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX > >& KinectCamera::jointIndicesForBones()
{
    if( KinectCamera::s_jointIndicesForBones.size() == 0 )
    {
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_SPINE ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_SPINE, NUI_SKELETON_POSITION_SHOULDER_CENTER ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_HEAD ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SHOULDER_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_SHOULDER_LEFT, NUI_SKELETON_POSITION_ELBOW_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_ELBOW_LEFT, NUI_SKELETON_POSITION_WRIST_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_WRIST_LEFT, NUI_SKELETON_POSITION_HAND_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SHOULDER_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_SHOULDER_RIGHT, NUI_SKELETON_POSITION_ELBOW_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_ELBOW_RIGHT, NUI_SKELETON_POSITION_WRIST_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_WRIST_RIGHT, NUI_SKELETON_POSITION_HAND_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_HIP_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_HIP_LEFT, NUI_SKELETON_POSITION_KNEE_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_KNEE_LEFT, NUI_SKELETON_POSITION_ANKLE_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_ANKLE_LEFT, NUI_SKELETON_POSITION_FOOT_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_HIP_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_HIP_RIGHT, NUI_SKELETON_POSITION_KNEE_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_KNEE_RIGHT, NUI_SKELETON_POSITION_ANKLE_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_ANKLE_RIGHT, NUI_SKELETON_POSITION_FOOT_RIGHT ) );
    }

    return s_jointIndicesForBones;
}

// static
const std::map< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX >, int >& KinectCamera::boneIndicesForJoints()
{
    if( s_boneIndicesForJoints.size() == 0 )
    {
        const auto& jifb = jointIndicesForBones();
        for( int b = 0; b < jifb.size(); ++b )
        {
            auto j0j1 = jifb[b];
            s_boneIndicesForJoints[ j0j1 ] = b;
        }
    }

    return s_boneIndicesForJoints;
}
#endif

// Allow asking for both RGBA and BGRA since we'll do the conversion for them.
bool isValidColorConfig( StreamConfig config )
{
    if( config.pixelFormat == PixelFormat::BGRA_U8888 ||
        config.pixelFormat == PixelFormat::BGR_U888 ||
        config.pixelFormat == PixelFormat::RGBA_U8888 ||
        config.pixelFormat == PixelFormat::RGB_U888 )
    {
        return
        (
            config.resolution == Vector2i{ 640, 480 } ||
            config.resolution == Vector2i{ 1280, 960 }
        );
    }
    // TODO(jiawen): YUV at 640x480 only.
    return false;
}

bool isValidDepthConfig( StreamConfig config )
{
    if( config.pixelFormat == PixelFormat::DEPTH_MM_U16 )
    {
        return
        (
            config.resolution == Vector2i{ 80, 60 } ||
            config.resolution == Vector2i{ 320, 240 } ||
            config.resolution == Vector2i{ 640, 480 }
        );
    }
    // TODO(jiawen): DEPTH_AND_PLAYER_INDEX at 320x240.
    return false;
}

bool isValidInfraredConfig( StreamConfig config )
{
    return
    (
        config.resolution == Vector2i{ 640, 480 } &&
        config.pixelFormat == PixelFormat::GRAY_U16
    );
}

NUI_IMAGE_RESOLUTION fromVector2i( const Vector2i& resolution )
{
    if( resolution == Vector2i{ 80, 60 } )
    {
        return NUI_IMAGE_RESOLUTION_80x60;
    }
    else if( resolution == Vector2i{ 320, 240 } )
    {
        return NUI_IMAGE_RESOLUTION_320x240;
    }
    else if( resolution == Vector2i{ 640, 480 } )
    {
        return NUI_IMAGE_RESOLUTION_640x480;
    }
    else if( resolution == Vector2i{ 1280, 960 } )
    {
        return NUI_IMAGE_RESOLUTION_1280x960;
    }
    else
    {
        return NUI_IMAGE_RESOLUTION_INVALID;
    }
}

KinectCamera::KinectCamera( const std::vector< StreamConfig >& streamConfig,
    int deviceIndex )
{
    // TODO: make a function to check for allowed combinations of streams and
    // expose it as a public interface.

    // Turn vector of StreamConfigs into flags:
    StreamConfig colorConfig;
    StreamConfig depthConfig;
    StreamConfig infraredConfig;

    for( int i = 0; i < static_cast< int >( streamConfig.size() ); ++i )
    {
        if( streamConfig[ i ].type == StreamType::COLOR &&
            isValidColorConfig( streamConfig[ i ] ) )
        {
            colorConfig = streamConfig[ i ];
            break;
        }
    }

    for( int i = 0; i < static_cast< int >( streamConfig.size() ); ++i )
    {
        if( streamConfig[ i ].type == StreamType::DEPTH &&
            isValidDepthConfig( streamConfig[ i ] ) )
        {
            depthConfig = streamConfig[ i ];
            break;
        }
    }

    for( int i = 0; i < static_cast< int >( streamConfig.size() ); ++i )
    {
        if( streamConfig[ i ].type == StreamType::INFRARED &&
            isValidInfraredConfig( streamConfig[ i ] ) )
        {
            infraredConfig = streamConfig[ i ];
            break;
        }
    }

    DWORD nuiFlags = 0;
    NUI_IMAGE_TYPE colorFormat = NUI_IMAGE_TYPE_COLOR;
    NUI_IMAGE_RESOLUTION colorResolution = NUI_IMAGE_RESOLUTION_INVALID;
    NUI_IMAGE_TYPE depthFormat = NUI_IMAGE_TYPE_DEPTH;
    NUI_IMAGE_RESOLUTION depthResolution = NUI_IMAGE_RESOLUTION_INVALID;
    if( colorConfig.type != StreamType::UNKNOWN )
    {
        nuiFlags |= NUI_INITIALIZE_FLAG_USES_COLOR;
        colorFormat = NUI_IMAGE_TYPE_COLOR;
        colorResolution = fromVector2i( colorConfig.resolution );
    }
    else if( infraredConfig.type != StreamType::UNKNOWN )
    {
        nuiFlags |= NUI_INITIALIZE_FLAG_USES_COLOR;
        colorFormat = NUI_IMAGE_TYPE_COLOR_INFRARED;
        colorResolution = fromVector2i( infraredConfig.resolution );
    }
    if( depthConfig.type != StreamType::UNKNOWN )
    {
        nuiFlags |= NUI_INITIALIZE_FLAG_USES_DEPTH;
        depthFormat = NUI_IMAGE_TYPE_DEPTH;
        depthResolution = fromVector2i( depthConfig.resolution );
    }

    m_impl = std::make_unique< KinectCameraImpl >( nuiFlags,
        colorFormat, colorResolution,
        depthFormat, depthResolution,
        deviceIndex
    );
}

KinectCamera::~KinectCamera()
{

}

bool KinectCamera::isValid() const
{
    return m_impl->isValid();
}

int KinectCamera::elevationAngle() const
{
    return m_impl->elevationAngle();
}

bool KinectCamera::setElevationAngle( int degrees )
{
    return m_impl->setElevationAngle( degrees );
}

bool KinectCamera::isNearModeEnabled() const
{
    return m_impl->isNearModeEnabled();
}

bool KinectCamera::setNearModeEnabled( bool b )
{
    return m_impl->setNearModeEnabled( b );
}

bool KinectCamera::setInfraredEmitterEnabled( bool b )
{
    return m_impl->setInfraredEmitterEnabled( b );
}

bool KinectCamera::rawAccelerometerReading( Vector4f& reading ) const
{
    return m_impl->rawAccelerometerReading( reading );
}

Vector3f KinectCamera::upVectorFromAccelerometer() const
{
    return m_impl->upVectorFromAccelerometer();
}

Vector2i KinectCamera::colorResolution() const
{
    return m_impl->colorResolution();
}

Intrinsics KinectCamera::colorIntrinsics() const
{
    return m_impl->colorIntrinsics();
}

Vector2i KinectCamera::depthResolution() const
{
    return m_impl->depthResolution();
}

Intrinsics KinectCamera::depthIntrinsics() const
{
    return m_impl->depthIntrinsics();
}

bool KinectCamera::pollOne( FrameView& frame,
    bool useExtendedDepth, int waitIntervalMilliseconds )
{
    return m_impl->pollOne(
        frame, useExtendedDepth, waitIntervalMilliseconds );
}

bool KinectCamera::pollAll( FrameView& frame,
    bool useExtendedDepth, int waitIntervalMilliseconds )
{
    return m_impl->pollAll(
        frame, useExtendedDepth, waitIntervalMilliseconds );
}

} } } // kinect1x, camera_wrappers libcgt
