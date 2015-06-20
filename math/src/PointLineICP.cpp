#include "PointLineICP.h"

#include <vecmath/Vector3f.h>

PointLineICP::PointLineICP( int maxNumIterations, float epsilon ) :

    m_maxNumIterations( maxNumIterations ),
    m_epsilon( epsilon ),

    m_A( 3, 3 ),
    m_b( 3, 1 )

{

}

bool PointLineICP::align( const std::vector< Vector2f >& srcPoints,
    const std::vector< Vector2f >& dstPoints, const std::vector< Vector2f >& dstNormals,
    const Matrix3f& initialGuess,
    Matrix3f& outputSrcToDestination )
{
    outputSrcToDestination = initialGuess;

    // make a copy of the source points that actually move
    std::vector< Vector2f > srcPoints2( srcPoints );

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
            Vector2f p = srcPoints2[ i ];
            Vector2f q = dstPoints[ i ];
            Vector2f n = dstNormals[ i ];

            float c = Vector2f::cross( p, n ).z;
            float d = Vector2f::dot( p - q, n );

            m_A( 0, 0 ) += c * c;
            m_A( 1, 0 ) += n.x * c;
            m_A( 2, 0 ) += n.y * c;

            m_A( 1, 1 ) += n.x * n.x;
            m_A( 2, 1 ) += n.y * n.x;

            m_A( 2, 2 ) += n.y * n.y;

            m_b[ 0 ] -= c * d;
            m_b[ 1 ] -= n.x * d;
            m_b[ 2 ] -= n.y * d;
        }

        FloatMatrix x = m_A.solveSPD( m_b, succeeded );

        if( succeeded )
        {
            float theta = x[0];
            float tx = x[1];
            float ty = x[2];

            Matrix3f incremental =
                Matrix3f::translation( tx, ty ) * Matrix3f::rotateZ( theta );

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

float PointLineICP::updateSourcePointsAndEvaluateEnergy( const Matrix3f& incremental,
    const std::vector< Vector2f >& dstPoints, const std::vector< Vector2f >& dstNormals,
    std::vector< Vector2f >& srcPoints2 )
{
    float energy = 0;

    for( size_t i = 0; i < srcPoints2.size(); ++i )
    {
        srcPoints2[i] = ( incremental * Vector3f( srcPoints2[i], 1 ) ).xy();

        float residual = Vector2f::dot( srcPoints2[i] - dstPoints[i], dstNormals[i] );
        energy += residual * residual;
    }

    return energy;
}