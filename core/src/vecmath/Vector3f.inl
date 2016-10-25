inline Vector3f::Vector3f()
{
    x = 0;
    y = 0;
    z = 0;
}

inline Vector3f::Vector3f( float f )
{
    x = f;
    y = f;
    z = f;
}

inline Vector3f::Vector3f( float _x, float _y, float _z )
{
    x = _x;
    y = _y;
    z = _z;
}

inline Vector3f::Vector3f( const Vector2f& _xy, float _z )
{
    xy = _xy;
    z = _z;
}

inline Vector3f::Vector3f( float _x, const Vector2f& _yz )
{
    x = _x;
    yz = _yz;
}

inline Vector3f::Vector3f( const Vector3d& v )
{
    x = static_cast< float >( v.x );
    y = static_cast< float >( v.y );
    z = static_cast< float >( v.z );
}

inline Vector3f::Vector3f( const Vector3i& v )
{
    x = static_cast< float >( v.x );
    y = static_cast< float >( v.y );
    z = static_cast< float >( v.z );
}

inline Vector3f& Vector3f::operator = ( const Vector3d& v )
{
    x = static_cast< float >( v.x );
    y = static_cast< float >( v.y );
    z = static_cast< float >( v.z );

    return *this;
}

inline Vector3f& Vector3f::operator = ( const Vector3i& v )
{
    x = static_cast< float >( v.x );
    y = static_cast< float >( v.y );
    z = static_cast< float >( v.z );

    return *this;
}

inline const float& Vector3f::operator [] ( int i ) const
{
    assert( i >= 0 );
    assert( i < 3 );
    return ( &x )[i];
}

inline float& Vector3f::operator [] ( int i )
{
    return ( &x )[i];
}


inline Vector2f Vector3f::xz() const
{
    return{ x, z };
}

inline Vector3f Vector3f::xyz() const
{
    return{ x, y, z };
}

inline Vector3f Vector3f::yzx() const
{
    return{ y, z, x };
}

inline Vector3f Vector3f::zxy() const
{
    return{ z, x, y };
}

inline float Vector3f::norm() const
{
    return sqrt( normSquared() );
}

inline float Vector3f::normSquared() const
{
    return( x * x + y * y + z * z );
}

inline void Vector3f::normalize()
{
    float rcpNorm = 1.0f / norm();
    x *= rcpNorm;
    y *= rcpNorm;
    z *= rcpNorm;
}

inline Vector3f Vector3f::normalized() const
{
    float rcpNorm = 1.0f / norm();
    return
    {
        x * rcpNorm,
        y * rcpNorm,
        z * rcpNorm
    };
}

inline Vector3f Vector3f::normalized( float& normOut ) const
{
    normOut = norm();
    float rcpNorm = 1.0f / normOut;
    return
    {
        x * rcpNorm,
        y * rcpNorm,
        z * rcpNorm
    };
}

inline void Vector3f::homogenize()
{
    if( z != 0 )
    {
        float rcpZ = 1.0f / z;
        x *= rcpZ;
        y *= rcpZ;
        z = 1;
    }
}

inline Vector3f Vector3f::homogenized() const
{
    if( z != 0 )
    {
        float rcpZ = 1.0f / z;
        return{ rcpZ * x, rcpZ * y, 1 };
    }
    else
    {
        return{ x, y, z };
    }
}

// static
inline float Vector3f::dot( const Vector3f& v0, const Vector3f& v1 )
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

// static
inline Vector3f Vector3f::cross( const Vector3f& v0, const Vector3f& v1 )
{
    return
    {
        v0.y * v1.z - v0.z * v1.y,
        v0.z * v1.x - v0.x * v1.z,
        v0.x * v1.y - v0.y * v1.x
    };
}

inline Vector3f::operator const float* () const
{
    return &x;
}

inline Vector3f::operator float* ()
{
    return &x;
}

inline std::string Vector3f::toString() const
{
    std::ostringstream sstream;
    sstream << "( " << x << ", " << y << ", " << z << ")";
    return sstream.str();
}

inline Vector3f& Vector3f::operator += ( const Vector3f& v )
{
    x += v.x;
    y += v.y;
    z += v.z;

    return *this;
}

inline Vector3f& Vector3f::operator -= ( const Vector3f& v )
{
    x -= v.x;
    y -= v.y;
    z -= v.z;

    return *this;
}

inline Vector3f& Vector3f::operator *= ( float f )
{
    x *= f;
    y *= f;
    z *= f;

    return *this;
}

inline Vector3f& Vector3f::operator /= ( float f )
{
    x /= f;
    y /= f;
    z /= f;

    return *this;
}

inline Vector3f operator + ( const Vector3f& v0, const Vector3f& v1 )
{
    return{ v0.x + v1.x, v0.y + v1.y, v0.z + v1.z };
}

inline Vector3f operator + ( const Vector3f& v, float f )
{
    return{ v.x + f, v.y + f, v.z + f };
}

inline Vector3f operator + ( float f, const Vector3f& v )
{
    return v + f;
}

inline Vector3f operator - ( const Vector3f& v0, const Vector3f& v1 )
{
    return{ v0.x - v1.x, v0.y - v1.y, v0.z - v1.z };
}

inline Vector3f operator - ( const Vector3f& v, float f )
{
    return{ v.x - f, v.y - f, v.z - f };
}

inline Vector3f operator - ( float f, const Vector3f& v )
{
    return{ f - v.x, f - v.y, f - v.z };
}

inline Vector3f operator - ( const Vector3f& v )
{
    return{ -v.x, -v.y, -v.z };
}

inline Vector3f operator * ( const Vector3f& v0, const Vector3f& v1 )
{
    return{ v0.x * v1.x, v0.y * v1.y, v0.z * v1.z };
}

inline Vector3f operator * ( float f, const Vector3f& v )
{
    return{ v.x * f, v.y * f, v.z * f };
}

inline Vector3f operator * ( const Vector3f& v, float f )
{
    return{ v.x * f, v.y * f, v.z * f };
}

inline Vector3f operator / ( const Vector3f& v, float f )
{
    return{ v.x / f, v.y / f, v.z / f };
}

inline Vector3f operator / ( const Vector3f& v0, const Vector3f& v1 )
{
    return{ v0.x / v1.x, v0.y / v1.y, v0.z / v1.z };
}

inline Vector3f operator / ( float f, const Vector3f& v )
{
    return{ f / v.x, f / v.y, f / v.z };
}

inline bool operator == ( const Vector3f& v0, const Vector3f& v1 )
{
    return( v0.x == v1.x && v0.y == v1.y && v0.z == v1.z );
}

inline bool operator != ( const Vector3f& v0, const Vector3f& v1 )
{
    return !( v0 == v1 );
}
