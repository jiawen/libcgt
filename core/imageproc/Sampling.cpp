#include "libcgt/core/imageproc/Sampling.h"

#include "libcgt/core/math/Arithmetic.h"
#include "libcgt/core/math/MathUtils.h"
#include "libcgt/core/vecmath/Vector2f.h"
#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Vector4f.h"

using namespace libcgt::core::math;

namespace
{
template< typename T >
T bilerp( Array2DReadView< T > view, float x, float y )
{
    // TODO: check this math, it can be simplified.
    // Use rect and 2d math.
    x = x - 0.5f;
    y = y - 0.5f;

    // clamp to edge
    x = clampToRangeInclusive( x, 0.f, static_cast< float >( view.width() ) );
    y = clampToRangeInclusive( y, 0.f, static_cast< float >( view.height() ) );

    int x0 = clampToRangeExclusive( floorToInt( x ), 0, view.width() );
    int x1 = clampToRangeExclusive( x0 + 1, 0, view.width() );
    int y0 = clampToRangeExclusive( floorToInt( y ), 0, view.height() );
    int y1 = clampToRangeExclusive( y0 + 1, 0, view.height() );

    float xf = x - x0;
    float yf = y - y0;

    T v00 = view[ { x0, y0 } ];
    T v01 = view[ { x0, y1 } ];
    T v10 = view[ { x1, y0 } ];
    T v11 = view[ { x1, y1 } ];

    T v0 = lerp( v00, v01, yf ); // x = 0
    T v1 = lerp( v10, v11, yf ); // x = 1

    return lerp( v0, v1, xf );
}
}

namespace libcgt { namespace core { namespace imageproc {

template< typename T >
T bilinearSample( Array2DReadView< T > view, const Vector2f& xy )
{
    return bilerp< T >( view, xy.x, xy.y );
}

// static
float bilinearSampleNormalized( Array2DReadView< float > view,
    const Vector2f& xy )
{
    return bilinearSample( view, xy * Vector2f( view.size() ) );
}

// Explicit instantiation.
template float bilinearSample< float >( Array2DReadView< float > view,
    const Vector2f& xy );
template Vector2f bilinearSample< Vector2f >(
    Array2DReadView< Vector2f > view, const Vector2f& xy );
template Vector3f bilinearSample< Vector3f >(
    Array2DReadView< Vector3f > view, const Vector2f& xy );
template Vector4f bilinearSample< Vector4f >(
    Array2DReadView< Vector4f > view, const Vector2f& xy );
template uint8x3 bilinearSample< uint8x3 >(
    Array2DReadView< uint8x3 > view, const Vector2f& xy );


} } } // imageproc, core, libcgt
