#pragma once

#include <math/MathUtils.h>
#include <vecmath/Matrix4f.h>
#include <vecmath/Range1f.h>
#include <vecmath/Range1i.h>

namespace libcgt { namespace core { namespace geometry {

// given an input range [inputMin, inputMax]
// and an output range [outputMin, outputMax]
//
// Computes the parameters scale and offset such that:
// for x \in [inputMin, inputMax]
// scale * x + offset is in the range [outputMin, outputMax].
// Call this transformBetween() just like for Rect2f.
void rescaleRangeToScaleOffset( float inputMin, float inputMax,
    float outputMin, float outputMax,
    float& scale, float& offset );

// Returns a Matrix4f that takes a point in the from range and
// maps it to a point in the to range with scale and translation only.
// from.origin --> to.origin
// from.origin + size --> to.origin + size
//
// p is represented as (x, y, 0, 1).
Matrix4f transformBetween( const Range1f& from, const Range1f& to );

float rescale( float x, const Range1f& src, const Range1f& dst );

int rescale( float x, const Range1f& src, const Range1i& dst );

float rescale( int x, const Range1i& src, const Range1f& dst );

int rescale( int x, const Range1i& src, const Range1i& dst );

} } } // geometry, core, libcgt

#include "RangeUtils.inl"
