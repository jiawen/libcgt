#include "geometry/BSpline2f.h"

#include <limits>

#include "math/MathUtils.h"

BSpline2f::BSpline2f() :

    m_basis
    (
        1.0f / 6.0f, -3.0f / 6.0f, 3.0f / 6.0f, -1.0f / 6.0f,
        4.0f / 6.0f,         0.0f,       -1.0f,  3.0f / 6.0f,
        1.0f / 6.0f,  3.0f / 6.0f, 3.0f / 6.0f, -3.0f / 6.0f,
        0.0f,         0.0f,        0.0f,  1.0f / 6.0f
    )
{

}

int BSpline2f::numControlPoints() const
{
    return static_cast< int >( m_controlPoints.size() );
}

const std::vector< Vector2f >& BSpline2f::controlPoints() const
{
    return m_controlPoints;
}

void BSpline2f::appendControlPoint( const Vector2f& p )
{
    m_controlPoints.push_back( p );
}

int BSpline2f::controlPointClosestTo( const Vector2f& p, float& distanceSquared ) const
{
    int index = -1;
    distanceSquared = std::numeric_limits< float >::max();

    for( int i = 0; i < numControlPoints(); ++i )
    {
        float d = ( m_controlPoints[ i ] - p ).normSquared();
        if( d < distanceSquared )
        {
            index = i;
            distanceSquared = d;
        }
    }

    return index;
}

void BSpline2f::moveControlPointTo( int i, const Vector2f& p )
{
    m_controlPoints[ i ] = p;
}

Vector2f BSpline2f::operator [] ( float t ) const
{
    return evaluateAt( t );
}

Vector2f BSpline2f::evaluateAt( float t ) const
{
    float u;
    int c0 = findControlPointStartIndex( t, u );

    Vector2f g0 = m_controlPoints[ c0 ];
    Vector2f g1 = m_controlPoints[ c0 + 1 ];
    Vector2f g2 = m_controlPoints[ c0 + 2 ];
    Vector2f g3 = m_controlPoints[ c0 + 3 ];

    float u2 = u * u;
    float u3 = u2 * u;

    Vector4f bu = m_basis * Vector4f( 1, u, u2, u3 );
    return
    {
        g0.x * bu.x + g1.x * bu.y + g2.x * bu.z + g3.x * bu.w,
        g0.y * bu.x + g1.y * bu.y + g2.y * bu.z + g3.y * bu.w
    };
}

Vector2f BSpline2f::tangentAt( float t ) const
{
    float u;
    int c0 = findControlPointStartIndex( t, u );

    Vector2f g0 = m_controlPoints[ c0 ];
    Vector2f g1 = m_controlPoints[ c0 + 1 ];
    Vector2f g2 = m_controlPoints[ c0 + 2 ];
    Vector2f g3 = m_controlPoints[ c0 + 3 ];

    Vector4f bu = m_basis * Vector4f( 0, 1, 2 * u, 3 * u * u );
    return
    {
        g0.x * bu.x + g1.x * bu.y + g2.x * bu.z + g3.x * bu.w,
        g0.y * bu.x + g1.y * bu.y + g2.y * bu.z + g3.y * bu.w
    };
}

Vector2f BSpline2f::normalAt( float t ) const
{
    return tangentAt( t ).normal();
}

Matrix3f BSpline2f::frameAt( float t ) const
{
    Vector2f p = evaluateAt( t );
    Vector2f tangent = tangentAt( t ).normalized();
    Vector2f normal = tangent.normal();

    return Matrix3f
    (
        normal.x, tangent.x, p.x,
        normal.y, tangent.y, p.y,
        0, 0, 1
    );
}

int BSpline2f::findControlPointStartIndex( float t, float& u ) const
{
    int n = numControlPoints();

    // control point parameterization:
    // t is uniformly sampled by control point index
    //
    // for example, for n = 6 control points
    // we have n - 3 = 3 segments:
    // each segment takes on a t range of length 1/3
    //
    // t \in [   0, 1/3 ], c = [ 0 1 2 3 ], map u to [0,1]
    // t \in [ 1/3, 2/3 ], c = [ 1 2 3 4 ], map u to [0,1]
    // t \in [ 2/3,   1 ], c = [ 2 3 4 5 ], map u to [0,1]

    float segmentLength = 1.0f / ( n - 3 );

    int c0 = static_cast< int >( t * ( n - 3 ) );
    c0 = MathUtils::clampToRangeExclusive( c0, 0, n - 3 );

    // where does t start for this segment?
    // c0 is the index:
    // the c0-th segment is of the range [ c0 * segmentLength, (c0 + 1 ) * segmentLength )

    float t0 = c0 * segmentLength;

    // remap [t0,t1] range to [0,1]
    float dt = t - t0;
    u = dt / segmentLength;

    return c0;
}
