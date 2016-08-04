#pragma once

#include <memory>

#include <common/Array2DView.h>
#include <common/BasicTypes.h>
#include <cameras/Intrinsics.h>
#include <vecmath/EuclideanTransform.h>

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
        Array2DView< uint16_t > infrared;
        // TODO(jiawen): YUV formats.

        // ----- Depth stream -----
        // Only one of packedDepth or (extendedDepth and playerIndex) will be
        // populated, depending on whether player tracking is enabled, and
        // whether the client requested packed or extended depth.
        bool depthUpdated;

        int64_t depthTimestamp;
        int depthFrameNumber;
        Array2DView< uint16_t > depth;
    };

    // TODO(jiawen): Useful configurations:
    // StreamConfig infrared{ Vector2i{ 640, 480 }, 30, ::PixelFormat::GRAY_U16 }
    // Range is [0, 1023].

    OpenNI2Camera(
        StreamConfig colorConfig =
            StreamConfig{ { 640, 480 }, 30, PixelFormat::RGB_U888 },
        StreamConfig depthConfig =
            StreamConfig{ { 640, 480 }, 30, PixelFormat::DEPTH_MM_U16 },
        StreamConfig infraredConfig = StreamConfig(),
        const char* uri = "" ); // "" means "any device"

    ~OpenNI2Camera();
    // TODO(VS2015): move constructor = default

    bool isValid() const;

    Intrinsics colorIntrinsics() const;
    Intrinsics depthIntrinsics() const;

    StreamConfig colorConfig() const;
    StreamConfig depthConfig() const;
    StreamConfig infraredConfig() const;

    void start();
    void stop();

    bool pollOne( Frame& frame, int timeoutMS = 0 );

    // Poll all registered streams.
    // Returns true if all succeeded.
    bool pollAll( Frame& frame, int timeoutMS = 0 );

private:

    std::unique_ptr< OpenNI2CameraImpl > m_impl;
};

} } } // openni2, camera_wrappers, libcgt
