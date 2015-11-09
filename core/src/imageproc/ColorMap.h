#pragma once

#include <common/BasicTypes.h>
#include <common/Array2DView.h>

class ColorMap
{
public:

    // TODO: const correct arrays

    // Given an input array "src",
    // clamps each pixel to [minValue, maxValue]
    // and maps it to the MATLAB "jet" pattern
    static void toJet( Array2DView< const float > src, float minValue, float maxValue,
        Array2DView< uint8x4 > dst );

};