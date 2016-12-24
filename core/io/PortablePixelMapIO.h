#pragma once

#include <string>

#include "libcgt/core/common/ArrayView.h"
#include "libcgt/core/common/BasicTypes.h"

class PortablePixelMapIO
{
public:

    // TODO: 16-bit

    static bool writeRGB( const std::string& filename,
        Array2DReadView< uint8x3 > image );
    static bool writeRGBText( const std::string& filename,
        Array2DReadView< uint8x3 > image );
};
