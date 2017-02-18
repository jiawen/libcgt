#pragma once

#include <cctype>
#include <cstdio>

#include "libcgt/core/common/Array2D.h"

class PortableGrayMapIO
{
public:

    struct PGMData
    {
        bool valid;

        int bitDepth; // 8 or 16
        int maxVal; // < 65536.

        Array2D< uint8_t > gray8;
        Array2D< uint16_t > gray16;
    };

    // TODO: does not parse comments
    // TODO: reads P5 (binary) only, P2 is the text version

    // The PGM format specifies that there is a maxVal in the header
    // which is > 0 and < 65536.
    // If maxVal < 256, then gray8 is populated and bitDepth = 8.
    // Else if maxVal < 65536, gray16 is populated and bitDepth = 16.
    // Otherwise, valid = false.
    static PGMData read( const std::string& filename );

    static bool writeBinary( Array2DReadView< uint8_t > image,
        const std::string& filename );
    static bool writeBinary( Array2DReadView< uint16_t > image,
        const std::string& filename );
};
