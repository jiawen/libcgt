#pragma once

#include <memory>
#include <vector>

#include <common/Array2DView.h>
#include <common/BasicTypes.h>
#include <cameras/Intrinsics.h>
#include <vecmath/EuclideanTransform.h>
#include <vecmath/Range1f.h>

#include <camera_wrappers/StreamConfig.h>

namespace libcgt { namespace camera_wrappers { namespace openni2 {

class OpenNI2CameraImpl;

class OpenNI2Camera
{
public:

    using Intrinsics = libcgt::core::cameras::Intrinsics;
    using EuclideanTransform = libcgt::core::vecmath::EuclideanTransform;

    // TODO(jiawen):
    // Consider changing things so that when polling, you should pass in owning
    // frames.

    // A simple (non-owning) container for a frame of data and its metadata
    // returned by the camera. When polling from the camera, "colorUpdated" or
    // "depthUpdated" will be set to true indicating which was updated.
    //
    // Timestamps are in microseconds from an arbitrary zero.
    struct Frame
    {
        // ----- Color stream -----
        // Only one of rgb or infrared will be populated, depending on which
        // color format was configured.
        bool colorUpdated;

        int64_t colorTimestamp;
        int colorFrameNumber;
        Array2DView< uint8x3 > rgb;
        // TODO(jiawen): YUV formats.

        bool infraredUpdated;
        int64_t infraredTimestamp;
        int infraredFrameNumber;
        Array2DView< uint16_t > infrared;

        // ----- Depth stream -----
        // Only one of packedDepth or (extendedDepth and playerIndex) will be
        // populated, depending on whether player tracking is enabled, and
        // whether the client requested packed or extended depth.
        bool depthUpdated;

        int64_t depthTimestamp;
        int depthFrameNumber;
        Array2DView< uint16_t > depth;
    };

    // 800 mm.
    static uint16_t minimumDepthMillimeters();

    // 4000 mm.
    static uint16_t maximumDepthMillimeters();

    // 0.8m - 4m.
    static Range1f depthRangeMeters();

    // TODO(jiawen): pass in resolution and scale.
    // Retrieve the "default" intrinsics that is approximately correct for all
    // models at 640x480.
    static Intrinsics defaultColorIntrinsics();

    // TODO(jiawen): pass in resolution and scale.
    // Retrieve the "default" intrinsics that is approximately correct for all
    // models at 640x480.
    static Intrinsics defaultDepthIntrinsics();

    // Get the transformation mapping: color_coord <-- depth_coord.
    // This transformation is in the OpenGL convention (y-up, z-out-of-screen).
    // Translation has units of millimeters.
    static EuclideanTransform colorFromDepthExtrinsicsMillimeters();

    // Get the transformation mapping: color_coord <-- depth_coord.
    // This transformation is in the OpenGL convention (y-up, z-out-of-screen).
    // Translation has units of millimeters.
    static EuclideanTransform colorFromDepthExtrinsicsMeters();

    // TODO(jiawen): Useful configurations:
    // StreamConfig infrared{ StreamType::INFRARED, Vector2i{ 640, 480 }, 30, ::PixelFormat::GRAY_U16 }
    // Range is [0, 1023].

    OpenNI2Camera( const std::vector< StreamConfig >& streamConfig =
        {
            {
                StreamType::COLOR,
                { 640, 480 }, PixelFormat::RGB_U888, 30,
                false
            },
            {
                StreamType::DEPTH,
                { 640, 480 }, PixelFormat::DEPTH_MM_U16, 30,
                false
            }
        },
        const char* uri = nullptr ); // nullptr means "any device"

    ~OpenNI2Camera();
    // TODO(VS2015): move constructor = default

    bool isValid() const;

    // Retrieve the factory-calibrated intrinsics for this device.
    Intrinsics colorIntrinsics() const;

    // Retrieve the factory-calibrated intrinsics for this device.
    Intrinsics depthIntrinsics() const;

    StreamConfig colorConfig() const;
    StreamConfig depthConfig() const;
    StreamConfig infraredConfig() const;

    void start();
    void stop();

    bool getAutoExposureEnabled();
    bool setAutoExposureEnabled( bool enabled );

    int getExposure();
    int getGain();
    bool setExposure( int exposure );
    bool setGain( int gain );

    bool getAutoWhiteBalanceEnabled();
    bool setAutoWhiteBalanceEnabled( bool enabled );

    bool pollColor( OpenNI2Camera::Frame& frame, int timeoutMS = 0 );

    bool pollDepth( OpenNI2Camera::Frame& frame, int timeoutMS = 0 );

    bool pollInfrared( OpenNI2Camera::Frame& frame, int timeoutMS = 0 );

    bool pollOne( Frame& frame, int timeoutMS = 0 );

    // Poll all registered streams.
    // Returns true if all succeeded.
    bool pollAll( Frame& frame, int timeoutMS = 0 );

private:

    std::unique_ptr< OpenNI2CameraImpl > m_impl;
};

} } } // openni2, camera_wrappers, libcgt
