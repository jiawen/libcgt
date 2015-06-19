#include "PointPlaneICP.h"

#include <vecmath/Vector4f.h>

PointPlaneICP::PointPlaneICP( int maxNumIterations, float epsilon ) :

	m_maxNumIterations( maxNumIterations ),
	m_epsilon( epsilon ),

	m_A( 6, 6 ),
	m_b( 6, 1 )

{

}

bool PointPlaneICP::align( const std::vector< Vector3f >& srcPoints,
	const std::vector< Vector3f >& dstPoints, const std::vector< Vector3f >& dstNormals,
	const Matrix4f& initialGuess,
	Matrix4f& outputSrcToDestination )
{
	outputSrcToDestination = initialGuess;

	// make a copy of the source points that actually move
	std::vector< Vector3f > srcPoints2( srcPoints );

	int itr = 0;
	bool succeeded = true;
	// TODO: could in theory pass in srcPoints, update it in one shot
	// one possibility is to multiply by the full every time, instead of each increment
	float energy = updateSourcePointsAndEvaluateEnergy( initialGuess,
		dstPoints, dstNormals,
		srcPoints2 );

	while( succeeded && // abort on singular configuration
		itr < m_maxNumIterations && // also exit if we
		energy > m_epsilon ) // exit loop if we found an acceptable minimum
	{		
		m_A.fill( 0 );
		m_b.fill( 0 );

		for( size_t i = 0; i < srcPoints2.size(); ++i )
		{
			Vector3f p = srcPoints2[ i ];
			Vector3f q = dstPoints[ i ];
			Vector3f n = dstNormals[ i ];

			Vector3f c = Vector3f::cross( p, n );
			float d = Vector3f::dot( p - q, n );

			m_A( 0, 0 ) += c.x * c.x;
			m_A( 1, 0 ) += c.y * c.x;
			m_A( 2, 0 ) += c.z * c.x;
			m_A( 3, 0 ) += n.x * c.x;
			m_A( 4, 0 ) += n.y * c.x;
			m_A( 5, 0 ) += n.z * c.x;
			
			m_A( 1, 1 ) += c.y * c.y;
			m_A( 2, 1 ) += c.z * c.y;
			m_A( 3, 1 ) += n.x * c.y;
			m_A( 4, 1 ) += n.y * c.y;
			m_A( 5, 1 ) += n.z * c.y;
			
			m_A( 2, 2 ) += c.z * c.z;
			m_A( 3, 2 ) += n.x * c.z;
			m_A( 4, 2 ) += n.y * c.z;
			m_A( 5, 2 ) += n.z * c.z;
			
			m_A( 3, 3 ) += n.x * n.x;
			m_A( 4, 3 ) += n.y * n.x;
			m_A( 5, 3 ) += n.z * n.x;
			
			m_A( 4, 4 ) += n.y * n.y;
			m_A( 5, 4 ) += n.z * n.y;
			
			m_A( 5, 5 ) += n.z * n.z;

			m_b[ 0 ] -= c.x * d;
			m_b[ 1 ] -= c.y * d;
			m_b[ 2 ] -= c.z * d;
			m_b[ 3 ] -= n.x * d;
			m_b[ 4 ] -= n.y * d;
			m_b[ 5 ] -= n.z * d;
		}

		FloatMatrix x = m_A.solveSPD( m_b, succeeded );

		if( succeeded )
		{
			float alpha = x[0];
			float beta = x[1];
			float gamma = x[2];
			float tx = x[3];
			float ty = x[4];
			float tz = x[5];

			Matrix4f incremental =
				Matrix4f::translation( tx, ty, tz ) *
				Matrix4f::rotateZ( gamma ) *
				Matrix4f::rotateY( beta ) *
				Matrix4f::rotateX( alpha );

			energy = updateSourcePointsAndEvaluateEnergy( incremental,
				dstPoints, dstNormals,
				srcPoints2 );
			
			// accumulate incremental transformation
			outputSrcToDestination = incremental * outputSrcToDestination;
		}
		printf( "At the end of iteration %d, energy = %f\n", itr, energy );

		++itr;
	}

	return succeeded;
}

float PointPlaneICP::updateSourcePointsAndEvaluateEnergy( const Matrix4f& incremental,
	const std::vector< Vector3f >& dstPoints, const std::vector< Vector3f >& dstNormals,
	std::vector< Vector3f >& srcPoints2 )
{
	float energy = 0;

	for( size_t i = 0; i < srcPoints2.size(); ++i )
	{
		srcPoints2[i] = ( incremental * Vector4f( srcPoints2[i], 1 ) ).xyz();

		float residual = Vector3f::dot( srcPoints2[i] - dstPoints[i], dstNormals[i] );
		energy += residual * residual;
	}

	return energy;
}