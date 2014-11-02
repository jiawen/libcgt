#pragma once

#include <common/BasicTypes.h>
#include <common/Array2D.h>

class QString;

class PNGIO
{
public:
    
    static bool writeRGB( QString filename, Array2DView< const uint8x3 > image );

    static bool writeRGBA( QString filename, Array2DView< const uint8x4 > image );
};