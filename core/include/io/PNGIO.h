#pragma once

#include <string>

#include <common/BasicTypes.h>
#include <common/Array2D.h>

class PNGIO
{
public:

    // TODO: implement uint16x3 / uint16x4 (16-bit) input.
    // TODO: needs big-endian input.

    static bool write( const std::string& filename, Array2DView< const uint8x3 > image );
    static bool write( const std::string& filename, Array2DView< const uint8x4 > image );
};