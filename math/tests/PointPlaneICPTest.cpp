#include <vector>

#include <math/MathUtils.h>
#include <math/Random.h>
#include <math/Sampling.h>

#include <vecmath/Matrix4f.h>

#include "PointPlaneICP.h"

void testPointPlaneICP()
{
    Random rnd( 0 );
    Matrix4f dstToSrc = Matrix4f::translation( 0.1f, -0.42f, 0.2f ) * Matrix4f::rotateX( MathUtils::degreesToRadians( 10.f ) );
    //Matrix4f dstToSrc = Matrix4f::rotateX( MathUtils::degreesToRadians( 10.f ) );
    //Matrix4f dstToSrc = Matrix4f::rotateX( MathUtils::degreesToRadians( 20.f ) );

    Matrix4f srcToDstGT = dstToSrc.inverse();

    int nPoints = 1024;

    std::vector< Vector3f > srcPoints( nPoints );
    std::vector< Vector3f > dstPoints( nPoints );
    std::vector< Vector3f > dstNormals( nPoints );

    for( size_t i = 0; i < dstPoints.size(); ++i )
    {
        dstPoints[i] = Vector3f( rnd.nextFloat(), rnd.nextFloat(), rnd.nextFloat() );
        dstNormals[i] = Sampling::areaSampleSphere( rnd.nextFloat(), rnd.nextFloat() );

        srcPoints[i] = dstToSrc.transformPoint( dstPoints[i] );
    }

    PointPlaneICP icp( 6, 0.01f );

    Matrix4f initialGuess = Matrix4f::identity();
    Matrix4f mSolution;
    bool succeeded = icp.align( srcPoints, dstPoints, dstNormals, initialGuess, mSolution );

    Matrix4f diff = srcToDstGT.inverse() - mSolution;
    diff.print();
}
