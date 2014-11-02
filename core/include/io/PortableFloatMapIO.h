#pragma once

#include <cstdint>
#include <vector>

#include <common/Array2D.h>
#include <common/Array2DView.h>

class QString;
class Vector3f;
class Vector4f;

class PortableFloatMapIO
{
public:

    // TODO: read all three types, return a variant.
    // TODO: get rid of QString
    // TODO: get rid of QFile:
    // read the PF header, then read width / height. If the second line only has one symbol
    // then read another line (sscanf returns 1 instead of 2).
    static Array2D< float > readGrayscale( QString filename );

    static Array2D< Vector3f > readRGB( QString filename );

    static Array2D< Vector4f > readRGBA( QString filename );

    // Writes a standard "PFM" format.
    // Header is "Pf" - grayscale.
	static bool writeGrayscale( QString filename, Array2DView< const float > image );

    // Writes a standard "PFM" format.
    // Header is "PF" - rgb.
	static bool writeRGB( QString filename, Array2DView< const Vector3f > image );
	
	// Writes a nonstandard "PFM4" format.
	// (header is "PF4", and includes an alpha channel)
	static bool writeRGBA( QString filename, Array2DView< const Vector4f > image );
};
