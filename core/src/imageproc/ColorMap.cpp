#include "imageproc/ColorMap.h"

#include <color/ColorUtils.h>
#include <math/MathUtils.h>

// static
void ColorMap::toJet( Array2DView< float > src, float minValue, float maxValue,
	Array2DView< ubyte4 > dst )
{
	for( int y = 0; y < src.height(); ++y )
	{
		for( int x = 0; x < src.width(); ++x )
		{
			float value = src( x, y );
			value = MathUtils::clampToRange( value, minValue, maxValue );
			float z = ( value - minValue ) / ( maxValue - minValue );

			dst( x, y ) = ColorUtils::floatToUnsignedByte( ColorUtils::colorMapJet( z ) );
		}
	}
}