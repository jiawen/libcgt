#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <cameras/Intrinsics.h>
#include <common/Array2D.h>
#include <common/BasicTypes.h>
#include <vecmath/EuclideanTransform.h>
#include <vecmath/Range1f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

#include <camera_wrappers/StreamConfig.h>

namespace libcgt { namespace camera_wrappers { namespace kinect1x {

class KinectCameraImpl;

class KinectCamera
{
public:

    using Intrinsics = libcgt::core::cameras::Intrinsics;
    using EuclideanTransform = libcgt::core::vecmath::EuclideanTransform;

    enum class Event
    {
        FAILED,
        TIMEOUT,
        OK
    };

    enum class Bone
    {
        LOWER_SPINE = 0,
        UPPER_SPINE,
        NECK,
        LEFT_CLAVICLE,
        LEFT_UPPER_ARM,
        LEFT_LOWER_ARM,
        LEFT_METACARPAL,
        RIGHT_CLAVICLE,
        RIGHT_UPPER_ARM,
        RIGHT_LOWER_ARM,
        RIGHT_METACARPAL,
        LEFT_HIP,
        LEFT_FEMUR,
        LEFT_LOWER_LEG,
        LEFT_METATARSUS,
        RIGHT_HIP,
        RIGHT_FEMUR,
        RIGHT_LOWER_LEG,
        RIGHT_METATARSUS,
        NUM_BONES
    };

    // A simple (non-owning) container for a frame of data and its metadata
    // returned by the camera. When polling from the camera, "colorUpdated",
    // "depthUpdated", and/or "skeletonUpdated" will be set to true indicating
    // which was updated.
    //
    // timestamps are milliseconds since the most recent NuiInitialize() call.
    struct Frame
    {
        // Only one of color or infrared will be populated, depending on which
        // color format was configured.
        bool colorUpdated;
        bool depthUpdated;
        bool infraredUpdated;

        // ----- Color stream -----
        int64_t colorTimestamp;
        int colorFrameNumber;
        // color will always have alpha set to 0.
        Array2DView< uint8x4 > color;
        // TODO(jiawen): YUV formats.

        // ----- Depth stream -----
        // Only one of packedDepth or (extendedDepth and playerIndex) will be
        // populated, depending on whether player tracking is enabled, and
        // whether the client requested packed or extended depth.

        int64_t depthTimestamp;
        int depthFrameNumber;
        bool depthCapturedWithNearMode; // Not updated if using packed depth.

        // ----- Infrared stream -----

        int64_t infraredTimestamp;
        int infraredFrameNumber;
        // The infrared stream has data only in the top 10 bits. Bits 5:0 are 0.
        Array2DView< uint16_t > infrared;

        // Each pixel of packedDepth is populated with a 16-bit value where:
        // - bits 15:3 is depth in millimeters.
        // - If player tracking is enabled, then bits 2:0 are set to the player
        //   index (1-6), or 0 for no player. If tracking is disabled, then
        //   bits 2:0 are set to 0.
        Array2DView< uint16_t > packedDepth;

        // Each pixel of extendedDepth is set to the full sensor depth range in
        // millimeters using the full 16-bit range (no bit shifting).
        Array2DView< uint16_t > extendedDepth;
        // If player tracking is enabled, then each pixel is set to the player
        // index (1-6), or 0 for no player.
        Array2DView< uint16_t > playerIndex;

        // TODO(jiawen): wrap skeleton.
        bool skeletonUpdated;
#if 0
        NUI_SKELETON_FRAME skeleton;
#endif
    };

    static int numDevices();

    // 800 mm.
    static uint16_t minimumDepthMillimeters();

    // 4000 mm.
    static uint16_t maximumDepthMillimeters();

    // 0.8m - 4m.
    static Range1f depthRangeMeters();

    // 400 mm.
    static uint16_t nearModeMinimumDepthMillimeters();

    // 3000 mm.
    static uint16_t nearModeMaximumDepthMillimeters();

    // 0.4m - 3m.
    static Range1f nearModeDepthRangeMeters();

    // The typical factory-calibrated intrinsics of the Kinect color camera.
    // Returns {{0, 0}, {0, 0}} for an invalid resolution.
    static Intrinsics colorIntrinsics( const Vector2i& resolution );

    // The typical factory-calibrated intrinsics of the Kinect depth camera.
    // Returns {{0, 0}, {0, 0}} for an invalid resolution.
    static Intrinsics depthIntrinsics( const Vector2i& resolution );

    // Get the transformation mapping: color_coord <-- depth_coord.
    // This transformation is in the OpenGL convention (y-up, z-out-of-screen).
    // Translation has units of millimeters.
    static EuclideanTransform colorFromDepthExtrinsicsMillimeters();

    // Get the transformation mapping: color_coord <-- depth_coord.
    // This transformation is in the OpenGL convention (y-up, z-out-of-screen).
    // Translation has units of millimeters.
    static EuclideanTransform colorFromDepthExtrinsicsMeters();

#if 0
    // Returns a vector of pairs of indices (i,j) such that within a
    // NUI_SKELETON_FRAME f, frame.SkeletonData[f].SkeletonPositions[i]
    // --> frame.SkeletonData[f].SkeletonPositions[j] is a bone.
    static const std::vector< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX > >& jointIndicesForBones();

    // The inverse of jointIndicesForBones().
    static const std::map< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX >, int >& boneIndicesForJoints();
#endif

    // Create a Kinect device.
    //
    // deviceIndex should be between [0, numDevices()),
    //
    // nuiFlags is an bit mask combination of:
    //   NUI_INITIALIZE_FLAG_USES_COLOR,
    //   NUI_INITIALIZE_FLAG_USES_DEPTH,
    //   NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX,
    //   NUI_INITIALIZE_FLAG_USES_SKELETON,
    //   NUI_INITIALIZE_FLAG_USES_AUDIO
    //
    //   Restrictions:
    //   - To enable skeleton tracking (NUI_INITIALIZE_FLAG_USES_SKELETON set),
    //     NUI_INITIALIZE_FLAG_USES_DEPTH or
    //     NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX must be set.
    //   Notes:
    //   - INFRARED images are imaged with the depth sensor, but arrive on the
    //     COLOR stream.
    //
    // colorResolution and depthResolution are self-explanatory.
    //
    ///  Restrictions:
    //   - colorResolution depends on the colorType:
    //     - NUI_IMAGE_TYPE_COLOR: 640x480 or 1280x960.
    //     - NUI_IMAGE_TYPE_COLOR_YUV: 640x480
    //     - NUI_IMAGE_TYPE_COLOR_RAW_YUV: 640x480
    //     - NUI_IMAGE_TYPE_COLOR_INFRARED: 640x480
    //   - depthResolution depends on whether player index tracking is enabled.
    //     - DEPTH: 80x60, 320x240, or 640x480.
    //     - DEPTH_AND_PLAYER_INDEX: 80,60 or 320x240.
    // TODO(jiawen): pass in extra flags for enabling player index and the
    // pixel format.
    KinectCamera(
        StreamConfig colorConfig =
            StreamConfig{ { 640, 480 }, 30, PixelFormat::RGB_U888, false },
        StreamConfig depthConfig =
            StreamConfig{ { 640, 480 }, 30, PixelFormat::DEPTH_MM_U16, false },
        StreamConfig infraredConfig = StreamConfig(),
        int deviceIndex = 0
    );

    KinectCamera( const KinectCamera& copy ) = delete;
    KinectCamera& operator = ( const KinectCamera& copy ) = delete;
    // TODO(VS2015): move constructor = default
    ~KinectCamera();

    // TODO(jiawen): store and provide accessors to stream configs.
    // TODO(jiawen): using that, allow mirror mode, RGB format.

    // Returns true if the camera is correctly initialized.
    bool isValid() const;

    // Get the current elevation angle in degrees.
    int elevationAngle() const;

    // Set the camera elevation angle. Returns whether it succeeded.
    bool setElevationAngle( int degrees );

    // Returns true if the depth stream is currently has near mode enabled.
    bool isNearModeEnabled() const;

    // Set the depth stream in near mode. Returns whether it succeeded.
    bool setNearModeEnabled( bool b );

    // Set b to true to enable the emitter,  false to disable to emitter.
    // Returns whether it succeeded.
    //
    // Note that this only works on the Kinect for Windows sensor (and not the
    // Kinect for Xbox 360 sensor).
    bool setInfraredEmitterEnabled( bool b );

    // returns the raw accelerometer reading
    // in units of g, with y pointing down and z pointing out of the camera
    // (w = 0)
    bool rawAccelerometerReading( Vector4f& reading ) const;

    // returns the camera "up" vector using the raw accelerometer reading
    // with y pointing up
    // equal to -rawAccelerometerReading.xyz().normalized()
    Vector3f upVectorFromAccelerometer() const;

    // The resolution of the configured color or infrared stream.
    // Returns (0, 0) if it's not configured.
    Vector2i colorResolution() const;

    // The typical factory-calibrated intrinsics of the Kinect color camera.
    // Returns {{0, 0}, {0, 0}} if not configured.
    Intrinsics colorIntrinsics() const;

    // The resolution of the configured depth stream.
    // Returns (0, 0) if it's not configured.
    Vector2i depthResolution() const;

    // The typical factory-calibrated intrinsics of the Kinect depth camera.
    // Returns {{0, 0}, {0, 0}} if not configured.
    Intrinsics depthIntrinsics() const;

    // TODO(jiawen): get rid of extended depth as a flag. Just have the client
    // pass in a non-null buffer.

    // Block until one of the input streams is available. This function blocks
    // for up to waitIntervalMilliseconds for data to be available. If
    // waitIntervalMilliseconds is 0, it will return immediately (whether data
    // is available or not). If waitIntervalMilliseconds is -1, it will wait
    // forever.
    //
    // The "updated" flag in "frame" corresponding to the channel will be set.
    bool pollOne( Frame& frame,
        bool useExtendedDepth = true,
        int waitIntervalMilliseconds = 0 );

    // Block until all the input streams are available. This function blocks
    // for up to waitIntervalMilliseconds for data to be available. If
    // waitIntervalMilliseconds is 0, it will return immediately (whether data
    // is available or not). If waitIntervalMilliseconds is -1, it will wait
    // forever.
    //
    // The "updated" flag in "frame" corresponding to the channels will be set.
    bool pollAll( Frame& frame,
        bool useExtendedDepth = true,
        int waitIntervalMilliseconds = 0 );

private:

    std::unique_ptr< KinectCameraImpl > m_impl;

};

} } } // kinect1x, camera_wrappers libcgt