#pragma once

#include <common/BasicTypes.h>
#include <common/Array2DView.h>

class QString;
class Vector3f;
class Vector4f;

class PortableFloatMapIO
{
public:

	static bool writeGrayscale( QString filename, Array2DView< float > image );

	static bool writeRGB( QString filename, Array2DView< Vector3f > image );
	
	// writes to a nonstandard "PFM4" format
	// (header is "PF4", and includes an alpha channel)
	static bool writeRGBA( QString filename, Array2DView< Vector4f > image );
};
