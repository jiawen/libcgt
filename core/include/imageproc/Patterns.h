#pragma once

#include "math/Random.h"
#include "vecmath/Vector4f.h"

#include "Image1f.h"
#include "Image4f.h"

class Patterns
{
public:

	static Image1f createCheckerboard( int width, int height, int checkerSize,
		float whiteColor = 1, float blackColor = 0 );		

	static Image4f createCheckerboard( int width, int height, int checkerSize,
		const Vector4f& whiteColor = Vector4f( 1.f, 1.f, 1.f, 1.f ),
		const Vector4f& blackColor = Vector4f( 0.8f, 0.8f, 0.8f, 1.f ) );

	Image1f createRandom( int width, int height, Random& random );
	Image4f createRandomFloat4( int width, int height, Random& random );

};
