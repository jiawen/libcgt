#pragma once

#include "Image4f.h"
#include "Image4ub.h"

class Compositing
{
public:

	// classic compositing operation:
	// C_o = a_f * C_f + ( 1 - a_f ) * C_b
	// a_o = a_f + a_b * ( 1 - a_f )
	// composites foreground over background into the buffer "composite" and returns it
	static Image4f compositeOver( const Image4f& foreground, const Image4f& background );

	// given the composite image "compositeRGBA"
	// and the foreground image "foregroundRGBA" (probably from matting)
	// divides out the alpha to extract the background color in "backgroundRGBA" and returns it
	static Image4f extractBackgroundColor( const Image4f& composite, const Image4f& foreground );
	static Image4ub extractBackgroundColor( const Image4ub& composite, const Image4ub& foreground );

private:

	static Vector4f extractBackgroundColor( const Vector4f& composite, const Vector4f& foreground );
};
