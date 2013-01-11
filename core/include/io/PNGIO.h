#pragma once

#include <common/BasicTypes.h>
#include <common/Array2DView.h>

class QString;

class PNGIO
{
public:

	static bool writeRGB( QString filename, Array2DView< ubyte3 > image );

	static bool writeRGBA( QString filename, Array2DView< ubyte4 > image );
};