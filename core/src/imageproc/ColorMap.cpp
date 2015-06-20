#include "imageproc/ColorMap.h"

#include <imageproc/ColorUtils.h>
#include <math/MathUtils.h>

using namespace libcgt::core::imageproc::colorutils;

// static
void ColorMap::toJet( Array2DView< const float > src, float minValue, float maxValue,
    Array2DView< uint8x4 > dst )
{
    for( int y = 0; y < src.height(); ++y )
    {
        for( int x = 0; x < src.width(); ++x )
        {
            float value = src[ { x, y } ];
            value = MathUtils::clampToRange( value, minValue, maxValue );
            float z = ( value - minValue ) / ( maxValue - minValue );

            dst[ { x, y } ] = toUInt8( colorMapJet( z ) );
        }
    }
}