#pragma once

#include <common/Array1DView.h>
#include <common/Array2DView.h>

class Vector2f;
class Vector3f;
class Vector4f;

namespace libcgt { namespace core { namespace imageproc {

// TODO:
// float linearSample( Array1DView< float > view, float x );

// Only valid for T = { float, Vector2f, Vector3f, Vector4f, uint8x3 }.
// x \in [0, width), y \in [0, height)
template< typename T >
T bilinearSample( Array2DView< const T > view, const Vector2f& xy );

// x and y in [0,1]
float bilinearSampleNormalized( Array2DView< const float > view,
    const Vector2f& xy );

} } } // imageproc, core, libcgt
