#include "libcgt/core/geometry/BSplineSurface3f.h"

#include <limits>

#include "libcgt/core/math/MathUtils.h"
#include "libcgt/core/vecmath/Vector3f.h"

using libcgt::core::math::clampToRangeExclusive;

BSplineSurface3f::BSplineSurface3f() :

    m_basis
    (
        1.0f / 6.0f, -3.0f / 6.0f, 3.0f / 6.0f, -1.0f / 6.0f,
        4.0f / 6.0f,         0.0f,       -1.0f,  3.0f / 6.0f,
        1.0f / 6.0f,  3.0f / 6.0f, 3.0f / 6.0f, -3.0f / 6.0f,
        0.0f,         0.0f,        0.0f,  1.0f / 6.0f
    )

{

}

int BSplineSurface3f::width() const
{
    return static_cast< int >( m_controlPoints.size() );
}

int BSplineSurface3f::height() const
{
    if( m_controlPoints.size() > 0 )
    {
        return static_cast< int >( m_controlPoints[0].size() );
    }
    else
    {
        return 0;
    }
}

Vector2i BSplineSurface3f::numControlPoints() const
{
    return{ width(), height() };
}

const std::vector< std::vector< Vector3f > >& BSplineSurface3f::controlPoints() const
{
    return m_controlPoints;
}

void BSplineSurface3f::appendControlPointRow( const std::vector< Vector3f >& row )
{
    // to handle the case of when we start off empty
    if( m_controlPoints.size() == 0 )
    {
        m_controlPoints.resize( row.size() );
    }

    int w = width();
    if( row.size() == w )
    {
        for( int x = 0; x < w; ++x )
        {
            m_controlPoints[x].push_back( row[x] );
        }
    }
}

void BSplineSurface3f::appendControlPointColumn( const std::vector< Vector3f >& column )
{
    if( m_controlPoints.size() == 0 || // to handle the case of when we start off empty
        column.size() == height() )
    {
        m_controlPoints.push_back( column );
    }
}

Vector2i BSplineSurface3f::controlPointClosestTo( const Vector3f& p, float& distanceSquared ) const
{
    Vector2i index{ -1, -1 };
    distanceSquared = std::numeric_limits< float >::max();

    for( int x = 0; x < width(); ++x )
    {
        for( int y = 0; y < height(); ++y )
        {
            float d = ( m_controlPoints[ x ][ y ] - p ).normSquared();
            if( d < distanceSquared )
            {
                index = { x, y };
                distanceSquared = d;
            }
        }
    }

    return index;
}

void BSplineSurface3f::moveControlPointTo( const Vector2i& ij, const Vector3f& p )
{
    m_controlPoints[ ij.x ][ ij.y ] = p;
}

Vector3f BSplineSurface3f::operator () ( float u, float v ) const
{
    float s;
    float t;
    Vector2i c0 = findControlPointStartIndex( u, v, s, t );

    Matrix4f gx;
    Matrix4f gy;
    Matrix4f gz;

    for( int y = 0; y < 4; ++y )
    {
        for( int x = 0; x < 4; ++x )
        {
            Vector3f controlPoint = m_controlPoints[ c0.x + x ][ c0.y + y ];
            gx( x, y ) = controlPoint.x;
            gy( x, y ) = controlPoint.y;
            gz( x, y ) = controlPoint.z;
        }
    }

    float s2 = s * s;
    float s3 = s2 * s;

    float t2 = t * t;
    float t3 = t2 * t;

    Vector4f bu = m_basis * Vector4f( 1, s, s2, s3 );
    Vector4f bv = m_basis * Vector4f( 1, t, t2, t3 );

    Vector4f b = bu * bv;

    float px = Vector4f::dot( bv, gx * bu );
    float py = Vector4f::dot( bv, gy * bu );
    float pz = Vector4f::dot( bv, gz * bu );

    return Vector3f( px, py, pz );
}

Vector2i BSplineSurface3f::findControlPointStartIndex( float u, float v, float& s, float& t ) const
{
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

    float segmentLengthU = 1.0f / ( width() - 3 );
    float segmentLengthV = 1.0f / ( height() - 3 );

    int c0 = static_cast< int >( u * ( width() - 3 ) );
    int d0 = static_cast< int >( v * ( height() - 3 ) );
    c0 = clampToRangeExclusive( c0, 0, width() - 3 );
    d0 = clampToRangeExclusive( d0, 0, height() - 3 );

    // where does t start for this segment?
    // c0 is the index:
    // the c0-th segment is of the range [ c0 * segmentLength, (c0 + 1 ) * segmentLength )

    float u0 = c0 * segmentLengthU;
    float v0 = d0 * segmentLengthV;

    // remap [t0,t1] range to [0,1]
    float ds = u - u0;
    float dt = v - v0;
    s = ds / segmentLengthU;
    t = dt / segmentLengthU;

    return{ c0, d0 };
}
