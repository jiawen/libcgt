#pragma once

#include <memory>
#include <vector>

#include "libcgt/core/common/ArrayView.h"
#include "libcgt/core/common/BasicTypes.h"
#include "libcgt/core/cameras/Intrinsics.h"
#include "libcgt/core/vecmath/EuclideanTransform.h"
#include "libcgt/core/vecmath/Range1f.h"

#include "libcgt/camera_wrappers/StreamConfig.h"

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
    // Timestamps are in nanoseconds from an arbitrary zero.
    struct FrameView
    {
        bool colorUpdated;
        bool infraredUpdated;
        bool depthUpdated;

        // ----- Color stream -----
        // Either rgb or infrared will be populated but not both. The driver
        // has a limitation where the color and infrared data are delivered
        // over the same stream and only one can be activate at a time.
        int64_t colorTimestampNS;
        int colorFrameNumber;
        Array2DWriteView< uint8x3 > color;
        // TODO(jiawen): YUV formats.

        int64_t infraredTimestampNS;
        int infraredFrameNumber;
        Array2DWriteView< uint16_t > infrared;

        // ----- Depth stream -----
        int64_t depthTimestampNS;
        int depthFrameNumber;
        Array2DWriteView< uint16_t > depth;
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
    // Note: the y-axis points up.
    static Intrinsics defaultColorIntrinsics();

    // TODO(jiawen): pass in resolution and scale.
    // Retrieve the "default" intrinsics that is approximately correct for all
    // models at 640x480.
    // Note: the y-axis points up.
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
                { 320, 240 }, PixelFormat::DEPTH_MM_U16, 30,
                false
            }
        },
        const char* uri = nullptr ); // nullptr means "any device"

    OpenNI2Camera( OpenNI2Camera&& move ) = default;
    OpenNI2Camera& operator = ( OpenNI2Camera&& move ) = default;
    ~OpenNI2Camera();

    OpenNI2Camera( const OpenNI2Camera& copy ) = delete;
    OpenNI2Camera& operator = ( const OpenNI2Camera& copy ) = delete;

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

    bool pollColor( FrameView& frame, int timeoutMS = 0 );

    bool pollDepth( FrameView& frame, int timeoutMS = 0 );

    bool pollInfrared( FrameView& frame, int timeoutMS = 0 );

    bool pollOne( FrameView& frame, int timeoutMS = 0 );

    // Poll all registered streams.
    // Returns true if all succeeded.
    bool pollAll( FrameView& frame, int timeoutMS = 0 );

private:

    std::unique_ptr< OpenNI2CameraImpl > m_impl;
};

} } } // openni2, camera_wrappers, libcgt
