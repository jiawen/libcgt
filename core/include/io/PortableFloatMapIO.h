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
	
	struct PFMData
	{
		bool valid;
		int nComponents;
		// scale < 0: little endian
		// scale > 0: big endian
		// abs( scale ): scaling factor between sample values
		//   and appropriate units, such as W/m^2.
		float scale;

		// Only one of these will be notNull().
		Array2D< float > grayscale;
		Array2D< Vector3f > rgb;
		Array2D< Vector4f > rgba;
	};

	// Reads a PFM file and returns a PFMData structure.
	// Depending on the format, only one of the Array2Ds will be notNull().
	// If the read fails, then all of them will be null and valid is set to false.
	static PFMData read( QString filename );

	// Writes a standard 3-component PFM format with header "Pf".
	static bool write( QString filename, Array2DView< float > image );

	// Writes a standard 3-component PFM format with header "PF".
	static bool write( QString filename, Array2DView< Vector3f > image );
	
	// Writes to a nonstandard 4-component "PFM4" format with header "PF4".
	// It's the same as PFM except it has 4 channels.
	static bool write( QString filename, Array2DView< Vector4f > image );
};
