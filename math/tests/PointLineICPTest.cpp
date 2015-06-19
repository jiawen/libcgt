#include <vector>

#include <math/MathUtils.h>
#include <math/Random.h>
#include <math/Sampling.h>

#include <vecmath/Matrix3f.h>

#include "PointLineICP.h"

void testPointLineICP()
{
	Random rnd( 0 );
	Matrix3f dstToSrc = Matrix3f::translation( 0.1f, -0.42f ) * Matrix3f::rotateZ( MathUtils::degreesToRadians( 10.f ) );
	//Matrix3f dstToSrc = Matrix3f::rotateZ( MathUtils::degreesToRadians( 10.f ) );
	//Matrix3f dstToSrc = Matrix3f::rotateZ( MathUtils::degreesToRadians( 20.f ) );

	Matrix3f srcToDstGT = dstToSrc.inverse();

	int nPoints = 1024;

	std::vector< Vector2f > srcPoints( nPoints );
	std::vector< Vector2f > dstPoints( nPoints );
	std::vector< Vector2f > dstNormals( nPoints );

	for( size_t i = 0; i < dstPoints.size(); ++i )
	{
		dstPoints[i] = Vector2f( rnd.nextFloat(), rnd.nextFloat() );
		dstNormals[i] = Sampling::perimeterSampleCircle( rnd.nextFloat() );

		srcPoints[i] = dstToSrc.transformPoint( dstPoints[i] );
	}

	PointLineICP icp( 6, 0.01f );

	Matrix3f initialGuess = Matrix3f::identity();
	Matrix3f mSolution;
	bool succeeded = icp.align( srcPoints, dstPoints, dstNormals, initialGuess, mSolution );

	Matrix3f diff = srcToDstGT - mSolution;
	diff.print();
}