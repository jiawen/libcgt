#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "vecmath/Vector4i.h"
#include "vecmath/Vector2i.h"
#include "vecmath/Vector3i.h"
#include "vecmath/Vector4f.h"

Vector4i::Vector4i()
{
    m_elements[ 0 ] = 0;
    m_elements[ 1 ] = 0;
    m_elements[ 2 ] = 0;
    m_elements[ 3 ] = 0;
}

Vector4i::Vector4i( int i )
{
    m_elements[ 0 ] = i;
    m_elements[ 1 ] = i;
    m_elements[ 2 ] = i;
    m_elements[ 3 ] = i;
}

Vector4i::Vector4i( int x, int y, int z, int w ) :
    x{ x },
    y{ y },
    z{ z },
    w{ w }
{

}

Vector4i::Vector4i( const Vector2i& xy, int z, int w )
{
    m_elements[0] = xy.x;
    m_elements[1] = xy.y;
    m_elements[2] = z;
    m_elements[3] = w;
}

Vector4i::Vector4i( int x, const Vector2i& yz, int w )
{
    m_elements[0] = x;
    m_elements[1] = yz.x;
    m_elements[2] = yz.y;
    m_elements[3] = w;
}

Vector4i::Vector4i( int x, int y, const Vector2i& zw )
{
    m_elements[0] = x;
    m_elements[1] = y;
    m_elements[2] = zw.x;
    m_elements[3] = zw.y;
}

Vector4i::Vector4i( const Vector2i& xy, const Vector2i& zw )
{
    m_elements[0] = xy.x;
    m_elements[1] = xy.y;
    m_elements[2] = zw.x;
    m_elements[3] = zw.y;
}

Vector4i::Vector4i( const Vector3i& xyz, int w )
{
    m_elements[0] = xyz.x;
    m_elements[1] = xyz.y;
    m_elements[2] = xyz.z;
    m_elements[3] = w;
}

Vector4i::Vector4i( int x, const Vector3i& yzw )
{
    m_elements[0] = x;
    m_elements[1] = yzw.x;
    m_elements[2] = yzw.y;
    m_elements[3] = yzw.z;
}

const int& Vector4i::operator [] ( int i ) const
{
    return m_elements[ i % 4 ];
}

int& Vector4i::operator [] ( int i )
{
    return m_elements[ i % 4 ];
}

Vector2i Vector4i::xy() const
{
    return{ x, y };
}

Vector2i Vector4i::yz() const
{
    return{ y, z };
}

Vector2i Vector4i::zw() const
{
    return{ z, w };
}

Vector2i Vector4i::wx() const
{
    return{ w, x };
}

Vector3i Vector4i::xyz() const
{
    return{ m_elements[0], m_elements[1], m_elements[2] };
}

Vector3i Vector4i::yzw() const
{
    return{ m_elements[1], m_elements[2], m_elements[3] };
}

Vector3i Vector4i::zwx() const
{
    return{ m_elements[2], m_elements[3], m_elements[0] };
}

Vector3i Vector4i::wxy() const
{
    return{ m_elements[3], m_elements[0], m_elements[1] };
}

Vector3i Vector4i::xyw() const
{
    return{ m_elements[0], m_elements[1], m_elements[3] };
}

Vector3i Vector4i::yzx() const
{
    return{ m_elements[ 1 ], m_elements[ 2 ], m_elements[ 0 ] };
}

Vector3i Vector4i::zwy() const
{
    return{ m_elements[ 2 ], m_elements[ 3 ], m_elements[ 1 ] };
}

Vector3i Vector4i::wxz() const
{
    return{ m_elements[ 3 ], m_elements[ 0 ], m_elements[ 2 ] };
}

float Vector4i::norm() const
{
    return sqrt( static_cast< float >( normSquared() ) );
}

int Vector4i::normSquared() const
{
    return
    (
        m_elements[ 0 ] * m_elements[ 0 ] +
        m_elements[ 1 ] * m_elements[ 1 ] +
        m_elements[ 2 ] * m_elements[ 2 ] +
        m_elements[ 3 ] * m_elements[ 3 ]
    );
}

Vector4f Vector4i::normalized() const
{
    float rLength = 1.f / norm();

    return Vector4f
    (
        rLength * m_elements[ 0 ],
        rLength * m_elements[ 1 ],
        rLength * m_elements[ 2 ],
        rLength * m_elements[ 3 ]
    );
}

void Vector4i::homogenize()
{
    if( m_elements[3] != 0 )
    {
        m_elements[0] /= m_elements[3];
        m_elements[1] /= m_elements[3];
        m_elements[2] /= m_elements[3];
        m_elements[3] = 1;
    }
}

Vector4i Vector4i::homogenized() const
{
    if( m_elements[3] != 0 )
    {
        return
        {
            m_elements[ 0 ] / m_elements[ 3 ],
            m_elements[ 1 ] / m_elements[ 3 ],
            m_elements[ 2 ] / m_elements[ 3 ],
            1
        };
    }
    else
    {
        return Vector4i
        {
            m_elements[ 0 ],
            m_elements[ 1 ],
            m_elements[ 2 ],
            m_elements[ 3 ]
        };
    }
}

void Vector4i::negate()
{
    m_elements[0] = -m_elements[0];
    m_elements[1] = -m_elements[1];
    m_elements[2] = -m_elements[2];
    m_elements[3] = -m_elements[3];
}

Vector4i& Vector4i::operator += ( const Vector4i& v )
{
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;

    return *this;
}

Vector4i& Vector4i::operator -= ( const Vector4i& v )
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;

    return *this;
}

Vector4i& Vector4i::operator *= ( int i )
{
    x *= i;
    y *= i;
    z *= i;
    w *= i;

    return *this;
}

Vector4i& Vector4i::operator /= ( int i )
{
    x /= i;
    y /= i;
    z /= i;
    w /= i;

    return *this;
}

Vector4i::operator const int* () const
{
    return m_elements;
}

Vector4i::operator int* ()
{
    return m_elements;
}

std::string Vector4i::toString() const
{
    std::ostringstream sstream;
    sstream << "( " << x << ", " << y << ", " << z << ", " << w << ")";
    return sstream.str();
}

// static
int Vector4i::dot( const Vector4i& v0, const Vector4i& v1 )
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z + v0.w * v1.w;
}

// static
Vector4f Vector4i::lerp( const Vector4i& v0, const Vector4i& v1, float alpha )
{
    return alpha * ( v1 - v0 ) + Vector4f( v0 );
}

//////////////////////////////////////////////////////////////////////////
// Operators
//////////////////////////////////////////////////////////////////////////

Vector4i operator + ( const Vector4i& v0, const Vector4i& v1 )
{
    return{ v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w };
}

Vector4i operator - ( const Vector4i& v0, const Vector4i& v1 )
{
    return{ v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w };
}

Vector4i operator * ( const Vector4i& v0, const Vector4i& v1 )
{
    return{ v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w };
}

Vector4i operator - ( const Vector4i& v )
{
    return{ -v.x, -v.y, -v.z, -v.w };
}

Vector4i operator * ( int c, const Vector4i& v )
{
    return{ c * v.x, c * v.y, c * v.z, c * v.w };
}

Vector4i operator * ( const Vector4i& v, int c )
{
    return{ c * v.x, c * v.y, c * v.z, c * v.w };
}

Vector4f operator * ( float f, const Vector4i& v )
{
    return Vector4f( f * v.x, f * v.y, f * v.z, f * v.w );
}

Vector4f operator * ( const Vector4i& v, float f )
{
    return Vector4f( f * v.x, f * v.y, f * v.z, f * v.w );
}

Vector4i operator / ( const Vector4i& v, int c )
{
    return{ v.x / c, v.y / c, v.z / c, v.w / c };
}
