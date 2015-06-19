#pragma once

#include <string>

#include <common/Array2D.h>
#include <common/Array2DView.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

class PortableFloatMapIO
{
public:

    struct PFMData
    {
        bool valid;
        int nComponents;
        float scale;

        Array2D< float > grayscale;
        Array2D< Vector3f > rgb;
        Array2D< Vector4f > rgba;
    };

    static PFMData read( const std::string& filename );

    // Writes a standard "PFM" format.
    // Header is "Pf" - grayscale.
    static bool write( const std::string& filename, Array2DView< const float > image );

    // Writes a standard "PFM" format.
    // Header is "PF" - rgb.
    static bool write( const std::string& filename, Array2DView< const Vector3f > image );
	
	// Writes a *nonstandard* "PFM4" format.
	// (header is "PF4", and includes an alpha channel)
    static bool write( const std::string& filename, Array2DView< const Vector4f > image );
};
