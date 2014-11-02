#pragma once

#include <common/BasicTypes.h>
#include <common/Array2DView.h>

class QString;
class Vector3f;

class PortablePixelMapIO
{
public:

	// TODO: text vs binary
	// TODO: can optimize this by checking stride for packed to write binary
	static bool writeRGB( QString filename, Array2DView< const uint8x3 > image );		

	// Image is clamped to [0,1] before writing.
	static bool writeRGB( QString filename, Array2DView< const Vector3f > image );
};
