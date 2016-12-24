#pragma once

#include <string>

#include "libcgt/core/common/ArrayView.h"
#include "libcgt/core/common/Array2D.h"
#include "libcgt/core/vecmath/Vector2f.h"
#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Vector4f.h"

class PortableFloatMapIO
{
public:

    struct PFMData
    {
        bool valid;
        int nComponents;
        float scale;

        Array2D< float > grayscale;
        Array2D< Vector2f > rg;
        Array2D< Vector3f > rgb;
        Array2D< Vector4f > rgba;
    };

    static PFMData read( const std::string& filename );

    // Writes a standard "PFM" format.
    // Header is "Pf" - grayscale.
    static bool write( const std::string& filename,
        Array2DReadView< float > image );

    // Writes a *nonstandard* "PFM2" format.
    // The header is "PF2" and is red and green only.
    static bool write( const std::string& filename,
        Array2DReadView< Vector2f > image );

    // Writes a standard "PFM" format.
    // Header is "PF" - rgb.
    static bool write( const std::string& filename,
        Array2DReadView< Vector3f > image );

    // Writes a *nonstandard* "PFM4" format.
    // The header is "PF4", and includes an alpha channel.
    static bool write( const std::string& filename,
        Array2DReadView< Vector4f > image );
};
