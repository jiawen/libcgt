#pragma once

#include <common/BasicTypes.h>
#include <common/ArrayView.h>
#include <vecmath/Range1f.h>
#include <vecmath/Vector4f.h>

namespace libcgt { namespace core { namespace imageproc {

// Given an input array "src", clamp each pixel to "srcRange", then maps it to
// and maps it to the MATLAB "jet" pattern.
void jet( Array2DReadView< float > src, const Range1f& srcRange,
    Array2DWriteView< uint8x4 > dst );

// Given an input array "src", clamp each pixel to "srcRange", then maps it to
// and maps it to the MATLAB "jet" pattern.
void jet( Array2DReadView< float > src, const Range1f& srcRange,
    Array2DWriteView< Vector4f > dst );

// Given an input array "src" of unit normals (where w = 0 means invalid), maps
// it to RGB = 0.5 * (normal + (1, 1, 1)). dst.alpha = src.w.
void normalsToRGBA( Array2DReadView< Vector4f > src,
    Array2DWriteView< uint8x4 > dst );

// Given an input array "src" of unit normals (where w = 0 means invalid), maps
// it to RGB = 0.5 * (normal + (1, 1, 1)). dst.alpha = src.w.
void normalsToRGBA( Array2DReadView< Vector4f > src,
    Array2DWriteView< Vector4f > dst );

void rgbToLuminance( Array2DReadView< uint8x3 > src,
    Array2DWriteView< uint8_t > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to
// [0, 255], then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DReadView< uint16_t > src,
    const Range1i& srcRange, const Range1i& dstRange,
    Array2DWriteView< uint8_t > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to
// [0, 255], then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DReadView< uint16_t > src,
    const Range1i& srcRange, const Range1i& dstRange,
    Array2DWriteView< uint8x3 > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to
// [0, 255], then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DReadView< uint16_t > src,
    const Range1i& srcRange, const Range1i& dstRange,
    uint8_t dstAlpha, Array2DWriteView< uint8x4 > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to [0,1],
// then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DReadView< float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DWriteView< uint8_t > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to [0,1],
// then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DReadView< float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DWriteView< uint8x4 > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to [0,1],
// then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DReadView< float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DWriteView< float > dst );

// Linearly remap every pixel in src from srcRange to dstRange, clamp to [0,1],
// then convert to a luminance value in dst.
void linearRemapToLuminance( Array2DReadView< float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DWriteView< Vector4f > dst );

} } } // imageproc, core, libcgt
