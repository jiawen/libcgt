#pragma once

#include <common/ArrayView.h>
#include <common/BasicTypes.h>

class Vector4f;

namespace libcgt { namespace core { namespace imageproc {

// Classical "over" operator:
// C_o = a_f * C_f + ( 1 - a_f ) * C_b
// a_o = a_f + a_b * ( 1 - a_f )
//
// Inputs must all be the same size.
void over( Array2DReadView< Vector4f > foreground,
    Array2DReadView< Vector4f > background,
    Array2DWriteView< Vector4f > output );

// Given the composite image "composite",
// and the foreground image "foreground" (probably from matting),
// divides out the alpha to extract the background color in "background".
void extractBackgroundColor( Array2DReadView< Vector4f > composite,
    Array2DReadView< Vector4f > foreground,
    Array2DWriteView< Vector4f > background );

// Given the composite image "composite",
// and the foreground image "foreground" (probably from matting),
// divides out the alpha to extract the background color in "background".
void extractBackgroundColor( Array2DReadView< uint8x4 > composite,
    Array2DReadView< uint8x4 > foreground,
    Array2DWriteView< uint8x4 > background );

// Given the composite color "composite",
// and the foreground color "foreground" (probably from matting),
// divides out the alpha to extract the background color.
Vector4f extractBackgroundColor( const Vector4f& composite, const Vector4f& foreground );

} } } // imageproc, core, libcgt
