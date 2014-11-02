#pragma once

#include <common/Array1DView.h>
#include <common/Array2DView.h>

class Vector2f;
class Vector3f;
class Vector4f;

class ImageSampling
{
public:

	// static float linearSample( Array1DView< float > view, float x );

	// x \in [0, width), y \in [0, height)
	static float bilinearSample( Array2DView< float > view, float x, float y );
	static Vector2f bilinearSample( Array2DView< Vector2f > view, float x, float y );
	static Vector3f bilinearSample( Array2DView< Vector3f > view, float x, float y );
	static Vector4f bilinearSample( Array2DView< Vector4f > view, float x, float y );

	// x and y in [0,1]
	static float bilinearSampleNormalized( Array2DView< float > view, float x, float y );

private:


};