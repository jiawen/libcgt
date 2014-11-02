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

	static bool write( QString filename, Array2DView< uint8x3 > image );		

    // The input is clamped to [0,1]	
	static bool write( QString filename, Array2DView< Vector3f > image );
};
