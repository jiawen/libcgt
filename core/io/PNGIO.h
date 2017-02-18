#pragma once

#include <string>

#include "libcgt/core/common/BasicTypes.h"
#include "libcgt/core/common/ArrayView.h"
#include "libcgt/core/common/Array2D.h"

class PNGIO
{
public:

    // TODO: implement uint16x3 / uint16x4 (16-bit) input.
    // TODO: LodePNG needs big-endian input.
    struct PNGData
    {
        bool valid;

        int bitDepth;
        int nComponents;

        Array2D< uint8_t > gray8;
        Array2D< uint8x2 > grayalpha8;
        Array2D< uint8x3 > rgb8;
        Array2D< uint8x4 > rgba8;

        Array2D< uint16_t > gray16;
        Array2D< uint16x2 > grayalpha16;
        Array2D< uint16x3 > rgb16;
        Array2D< uint16x4 > rgba16;
    };

    static PNGData read( const std::string& filename );

    static bool write( Array2DReadView< uint8_t > image,
        const std::string& filename );
    static bool write( Array2DReadView< uint8x3 > image,
        const std::string& filename );
    static bool write( Array2DReadView< uint8x4 > image,
        const std::string& filename );

    static bool write( Array2DReadView< uint16_t > image,
        const std::string& filename );
};
