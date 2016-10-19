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

// static
inline float Vector3f::dot( const Vector3f& v0, const Vector3f& v1 )
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

// static
inline Vector3f Vector3f::cross( const Vector3f& v0, const Vector3f& v1 )
{
    return Vector3f
    (
        v0.y * v1.z - v0.z * v1.y,
        v0.z * v1.x - v0.x * v1.z,
        v0.x * v1.y - v0.y * v1.x
    );
}

inline Vector3f::operator const float* () const
{
    return &x;
}

inline Vector3f::operator float* ()
{
    return &x;
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
    return Vector3f( v0.x + v1.x, v0.y + v1.y, v0.z + v1.z );
}

inline Vector3f operator - ( const Vector3f& v0, const Vector3f& v1 )
{
    return Vector3f( v0.x - v1.x, v0.y - v1.y, v0.z - v1.z );
}

inline Vector3f operator - ( const Vector3f& v )
{
    return Vector3f( -v.x, -v.y, -v.z );
}

inline Vector3f operator * ( const Vector3f& v0, const Vector3f& v1 )
{
    return Vector3f( v0.x * v1.x, v0.y * v1.y, v0.z * v1.z );
}

inline Vector3f operator * ( float f, const Vector3f& v )
{
    return Vector3f( v.x * f, v.y * f, v.z * f );
}

inline Vector3f operator * ( const Vector3f& v, float f )
{
    return Vector3f( v.x * f, v.y * f, v.z * f );
}

inline Vector3f operator / ( const Vector3f& v, float f )
{
    return Vector3f( v.x / f, v.y / f, v.z / f );
}

inline Vector3f operator / ( const Vector3f& v0, const Vector3f& v1 )
{
    return Vector3f( v0.x / v1.x, v0.y / v1.y, v0.z / v1.z );
}

inline Vector3f operator / ( float f, const Vector3f& v )
{
    return Vector3f( f / v.x, f / v.y, f / v.z );
}

inline bool operator == ( const Vector3f& v0, const Vector3f& v1 )
{
    return( v0.x == v1.x && v0.y == v1.y && v0.z == v1.z );
}

inline bool operator != ( const Vector3f& v0, const Vector3f& v1 )
{
    return !( v0 == v1 );
}
