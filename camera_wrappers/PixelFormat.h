#pragma once

#include <cstdint>

namespace libcgt { namespace camera_wrappers {

enum class PixelFormat : uint32_t
{
    INVALID = 0,

    // Metric Depth maps.
    DEPTH_MM_U16 = 1,           // Depth in millimeters, 16-bit unsigned.
    DEPTH_M_F32 = 2,            // Depth in meters, 32-bit float.
    DEPTH_M_F16 = 3,            // Depth in meters, 16-bit float.

    RGBA_U8888 = 16,
    RGB_U888 = 17,
    BGRA_U8888 = 18,
    BGR_U888 = 19,

    GRAY_U8 = 128,
    GRAY_U16 = 129,
    GRAY_U32 = 130,

    // Uncalibrated disparity maps (from stereo).
    // These will need to be converted to metric depth to be rendered:
    // depth = (f * b) / disparity. f should be in the same units as disparity
    // (e.g., pixels), and b should be in metric units (e.g., meters).
    DISPARITY_S8 = 256,         // Disparity without units, 8-bit signed int.
    DISPARITY_S16 = 257,        // Disparity without units, 16-bit signed int.
    DISPARITY_S32 = 258,        // Disparity without units, 32-bit signed int.
    DISPARITY_F16 = 259,        // Disparity without units, 16-bit float.
    DISPARITY_F32 = 260,        // Disparity without units, 32-bit float.

    // Uncalibrated depth maps (from stereo or structure from motion).
    // These will need to be converted to metric depth to be rendered:
    // depth_metric = scale * depth_uncalibrated. scale should have units of
    // (meters / units_of_image_content).
    DEPTH_UNCALIBRATED_S8 = 288,   // Depth without units, 8-bit signed int.
    DEPTH_UNCALIBRATED_U8 = 289,   // Depth without units, 8-bit unsigned int.
    DEPTH_UNCALIBRATED_S16 = 290,  // Depth without units, 16-bit signed int.
    DEPTH_UNCALIBRATED_U16 = 291,  // Depth without units, 16-bit unsigned int.
    DEPTH_UNCALIBRATED_S32 = 292,  // Depth without units, 32-bit signed int.
    DEPTH_UNCALIBRATED_U32 = 293,  // Depth without units, 32-bit unsigned int.
    DEPTH_UNCALIBRATED_F16 = 294,  // Depth without units, 16-bit float.
    DEPTH_UNCALIBRATED_F32 = 295   // Depth without units, 32-bit float.
};

// Get the size of one pixel in the given format, in bytes.
// Return zero for INVALID or an unknown format.
uint32_t pixelSizeBytes( PixelFormat format );

} } // camera_wrappers, libcgt
