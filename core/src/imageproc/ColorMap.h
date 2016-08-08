#pragma once

#include <common/BasicTypes.h>
#include <common/Array2DView.h>
#include <vecmath/Range1f.h>
#include <vecmath/Vector4f.h>

namespace libcgt { namespace core { namespace imageproc {

// Given an input array "src", clamp each pixel to "srcRange", then maps it to
// and maps it to the MATLAB "jet" pattern.
void jet( Array2DView< const float > src, const Range1f& srcRange,
    Array2DView< uint8x4 > dst );

// Given an input array "src", clamp each pixel to "srcRange", then maps it to
// and maps it to the MATLAB "jet" pattern.
void jet( Array2DView< const float > src, const Range1f& srcRange,
    Array2DView< Vector4f > dst );

// Given an input array "src" of unit normals (where w = 0 means invalid), maps
// it to RGB = 0.5 * (normal + (1, 1, 1)). dst.alpha = src.w.
void normalsToRGBA( Array2DView< const Vector4f > src,
    Array2DView< uint8x4 > dst );

// Given an input array "src" of unit normals (where w = 0 means invalid), maps
// it to RGB = 0.5 * (normal + (1, 1, 1)). dst.alpha = src.w.
void normalsToRGBA( Array2DView< const Vector4f > src,
    Array2DView< Vector4f > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to
// [0, 255], then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DView< const uint16_t > src,
    const Range1i& srcRange, const Range1i& dstRange,
    Array2DView< uint8_t > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to
// [0, 255], then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DView< const uint16_t > src,
    const Range1i& srcRange, const Range1i& dstRange,
    Array2DView< uint8x3 > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to
// [0, 255], then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DView< const uint16_t > src,
    const Range1i& srcRange, const Range1i& dstRange,
    uint8_t dstAlpha, Array2DView< uint8x4 > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to [0,1],
// then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DView< const float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DView< uint8_t > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to [0,1],
// then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DView< const float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DView< uint8x4 > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to [0,1],
// then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DView< const float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DView< float > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to [0,1],
// then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DView< const float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DView< Vector4f > dst );

} } } // imageproc, core, libcgt
