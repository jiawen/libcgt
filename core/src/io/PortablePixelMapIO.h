#pragma once

#include <string>

#include <common/ArrayView.h>
#include <common/BasicTypes.h>

class PortablePixelMapIO
{
public:

    // TODO: 16-bit

    static bool writeRGB( const std::string& filename,
        Array2DReadView< uint8x3 > image );
    static bool writeRGBText( const std::string& filename,
        Array2DReadView< uint8x3 > image );
};
