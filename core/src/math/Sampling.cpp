#include "math/Sampling.h"

#include <algorithm>
#include <cmath>

#include "math/MathUtils.h"
#include "math/Random.h"
#include "math/SamplingPatternND.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
void Sampling::latinHypercubeSampling( Random& random, SamplingPatternND* pPattern )
{
	int nDimensions = pPattern->getNumDimensions();
	int nSamples = pPattern->getNumSamples();
	float* samples = pPattern->getSamples();

	// generate samples along diagonal boxes
	float delta = 1.0f / nSamples;
	for( int i = 0; i < nSamples; ++i )
	{
		for( int j = 0; j < nDimensions; ++j )
		{
			samples[ i * nDimensions + j ] = ( i + random.nextFloat() ) * delta;
		}
	}

	// permute samples in each dimension
	for( int i = 0; i < nDimensions; ++i )
	{
		for( int j = 0; j < nSamples; ++j )
		{
			int otherSample = random.nextIntInclusive( nSamples - 1 );
			swap( samples[ j * nDimensions + i ], samples[ otherSample * nDimensions + i ] );
		}
	}
}

// static
Vector2f Sampling::areaSampleDisc( float u0, float u1 )
{
	float r = sqrt( u0 );
	float theta = MathUtils::TWO_PI * u1;
	return Vector2f( r * cos( theta ), r * sin( theta ) );
}

// static
Vector2f Sampling::concentricSampleDisc( float u0, float u1 )
{
	float r;
	float theta;

	// Map uniform random numbers to [-1, 1]^2
	float sx = 2 * u0 - 1;
	float sy = 2 * u1 - 1;

	// Map square to (r, theta)
	// Handle degeneracy at the origin
	if( sx == 0.0 && sy == 0.0 )
	{
		return Vector2f( 0, 0 );
	}
	
	if( sx >= -sy )
	{
		if( sx > sy )
		{
			// Handle first region of disk
			r = sx;
			if( sy > 0.0 )
			{
				theta = sy / r;
			}
			else
			{
				theta = 8.0f + sy / r;
			}
		}
		else
		{
			// Handle second region of disk
			r = sy;
			theta = 2.0f - sx / r;
		}
	}
	else
	{
		if( sx <= sy )
		{
			// Handle third region of disk
			r = -sx;
			theta = 4.0f - sy / r;
		}
		else
		{
			// Handle fourth region of disk
			r = -sy;
			theta = 6.0f + sx / r;
		}
	}
	
	theta *= MathUtils::QUARTER_PI;
	return Vector2f( r * cos( theta ), r * sin( theta ) );
}

// static
Vector3f Sampling::areaSampleSphere( float u0, float u1 )
{
	// See: http://www.cs.cmu.edu/~mws/rpos.html
	float z = 2 * u0 - 1; // [-1,1]
	float phi = MathUtils::TWO_PI * u1; // [0,2*pi];

	float sqrtOneMinusSSquared = sqrt( 1.0f - z * z );
	float cosPhi = cos( phi );
	float sinPhi = sin( phi );

	return Vector3f
	(
		sqrtOneMinusSSquared * cosPhi,
		sqrtOneMinusSSquared * sinPhi,
		z
	);
}

// static
Vector3f Sampling::areaSampleTriangle( float u0, float u1 )
{
	if( u0 + u1 > 1 )
	{
		u0 = 1 - u0;
		u1 = 1 - u1;
	}

	return Vector3f( 1 - u0 - u1, u0, u1 );
}
