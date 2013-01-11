#pragma once

#include <common/BasicTypes.h>
#include <common/Array2DView.h>

class ColorMap
{
public:

	// Given an imput array
	// clamps it to [minValue, maxValue]
	// and maps it to the MATLAB "jet" pattern
	static void toJet( Array2DView< float > src, float minValue, float maxValue,
		Array2DView< ubyte4 > dst );

};