#pragma once

#include <string>

#include <common/BasicTypes.h>
#include <common/Array2DView.h>

class PortablePixelMapIO
{
public:

    static bool writeRGB( const std::string& filename, Array2DView< const uint8x3 > image );
    static bool writeRGBText( const std::string& filename, Array2DView< const uint8x3 > image );
};
