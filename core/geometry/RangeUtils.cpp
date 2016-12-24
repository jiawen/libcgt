#include "libcgt/core/geometry/RangeUtils.h"

namespace libcgt { namespace core { namespace geometry {

void rescaleRangeToScaleOffset( float inputMin, float inputMax,
    float outputMin, float outputMax,
    float& scale, float& offset )
{
    float inputRange = inputMax - inputMin;
    float outputRange = outputMax - outputMin;

    // y = outputMin + [ ( x - inputMin ) / inputRange ] * outputRange
    //   = outputMin + ( x * outputRange / inputRange ) - ( inputMin * outputRange / inputRange )
    //
    // -->
    //
    // scale = outputRange / inputRange
    // offset = outputMin - inputMin * outputRange / inputRange

    scale = outputRange / inputRange;
    offset = outputMin - inputMin * scale;
}

Matrix4f transformBetween( const Range1f& from, const Range1f& to )
{
    return Matrix4f::translation( { to.origin, 0.0f, 0.0f } ) *
        Matrix4f::scaling( { to.size, 1.0f, 1.0f } ) *
        Matrix4f::scaling( { 1.0f / from.size, 1.0f, 1.0f } ) *
        Matrix4f::translation( { -from.origin, 0.0f, 0.0f } );
}

} } } // geometry, core, libcgt
