#pragma once

#include <common/BasicTypes.h>
#include <common/Array2DView.h>

class QString;
class Vector3f;

class PortablePixelMapIO
{
public:

	// TODO: text vs binary

	/*
	// scale > 0 is big endian, < 0 is little endian
	static bool writeRGB( QString filename,
		Vector3f* avRGB,
		int width, int height,
		float scale = -1.0f );
		*/

	// TODO: can optimize this by checking stride for packed
	static bool writeRGB( QString filename, Array2DView< ubyte3 > image );		

	// elements of image needs to be in [0,1]^3
	static bool writeRGB( QString filename, Array2DView< Vector3f > image );
};
