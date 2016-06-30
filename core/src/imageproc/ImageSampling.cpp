#include "imageproc/ImageSampling.h"

#include <math/Arithmetic.h>
#include <math/MathUtils.h>

#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

using libcgt::core::math::clampToRangeInclusive;
using libcgt::core::math::clampToRangeExclusive;
using libcgt::core::math::floorToInt;
using libcgt::core::math::lerp;

// static
template< typename T >
T bilerp( Array2DView< T > view, float x, float y )
{
    // TODO(jiawen): check this math, it can be simplified.
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

    float xf = x - ( x0 + 0.5f );
    float yf = y - ( y0 + 0.5f );

    T v00 = view[ { x0, y0 } ];
    T v01 = view[ { x0, y1 } ];
    T v10 = view[ { x1, y0 } ];
    T v11 = view[ { x1, y1 } ];

    T v0 = lerp( v00, v01, yf ); // x = 0
    T v1 = lerp( v10, v11, yf ); // x = 1

    return lerp( v0, v1, xf );
}

float ImageSampling::bilinearSample( Array2DView< float > view, float x, float y )
{
    return bilerp< float >( view, x, y );
}

Vector2f ImageSampling::bilinearSample( Array2DView< Vector2f > view, float x, float y )
{
    return bilerp< Vector2f >( view, x, y );
}

Vector3f ImageSampling::bilinearSample( Array2DView< Vector3f > view, float x, float y )
{
    return bilerp< Vector3f >( view, x, y );
}

Vector4f ImageSampling::bilinearSample( Array2DView< Vector4f > view, float x, float y )
{
    return bilerp< Vector4f >( view, x, y );
}

// static
float ImageSampling::bilinearSampleNormalized( Array2DView< float > view, float x, float y )
{
    return bilinearSample( view, x * view.width(), y * view.height() );
}
