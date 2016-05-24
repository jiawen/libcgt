#include "imageproc/ColorMap.h"

#include <imageproc/ColorUtils.h>
#include <math/MathUtils.h>

using libcgt::core::imageproc::colorutils::toUInt8;
using libcgt::core::imageproc::colorutils::colorMapJet;
using libcgt::core::math::clampToRange;

// static
void ColorMap::toJet( Array2DView< const float > src, float minValue, float maxValue,
    Array2DView< uint8x4 > dst )
{
    for( int y = 0; y < src.height(); ++y )
    {
        for( int x = 0; x < src.width(); ++x )
        {
            float value = src[ { x, y } ];
            value = clampToRange(value, minValue, maxValue);
            float z = ( value - minValue ) / ( maxValue - minValue );

            dst[ { x, y } ] = toUInt8( colorMapJet( z ) );
        }
    }
}
