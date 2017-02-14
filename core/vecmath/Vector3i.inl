inline Vector3i::Vector3i()
{
    m_elements[ 0 ] = 0;
    m_elements[ 1 ] = 0;
    m_elements[ 2 ] = 0;
}

inline Vector3i::Vector3i( int i )
{
    m_elements[ 0 ] = i;
    m_elements[ 1 ] = i;
    m_elements[ 2 ] = i;
}

inline Vector3i::Vector3i( int x, int y, int z )
{
    m_elements[ 0 ] = x;
    m_elements[ 1 ] = y;
    m_elements[ 2 ] = z;
}

inline Vector3i::Vector3i( const Vector2i& xy, int z )
{
    m_elements[0] = xy.x;
    m_elements[1] = xy.y;
    m_elements[2] = z;
}

inline Vector3i::Vector3i( int x, const Vector2i& yz )
{
    m_elements[0] = x;
    m_elements[1] = yz.x;
    m_elements[2] = yz.y;
}

inline const int& Vector3i::operator [] ( int i ) const
{
    return m_elements[ i ];
}

inline int& Vector3i::operator [] ( int i )
{
    return m_elements[ i ];
}

inline Vector2i Vector3i::zx() const
{
    return{ z, x };
}

inline Vector2i Vector3i::yx() const
{
    return{ y, x };
}

inline Vector2i Vector3i::zy() const
{
    return{ z, y };
}

inline Vector2i Vector3i::xz() const
{
    return{ x, z };
}

inline Vector3i Vector3i::xyz() const
{
    return{ x, y, z };
}

inline Vector3i Vector3i::yzx() const
{
    return{ y, z, x };
}

inline Vector3i Vector3i::zxy() const
{
    return{ z, x, y };
}

inline float Vector3i::norm() const
{
    return sqrt( static_cast< float >( normSquared() ) );
}

inline int Vector3i::normSquared() const
{
    return( x * x + y * y + z * z );
}

inline Vector3f Vector3i::normalized() const
{
    float rcpNorm = 1.f / norm();
    return Vector3f
    (
        rcpNorm * x,
        rcpNorm * y,
        rcpNorm * z
    );
}

inline Vector3i::operator const int* () const
{
    return m_elements;
}

inline Vector3i::operator int* ()
{
    return m_elements;
}

inline std::string Vector3i::toString() const
{
    std::ostringstream sstream;
    sstream << "( " << x << ", " << y << ", " << z << " )";
    return sstream.str();
}

// static
inline int Vector3i::dot( const Vector3i& v0, const Vector3i& v1 )
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

// static
inline Vector3i Vector3i::cross( const Vector3i& v0, const Vector3i& v1 )
{
    return
    {
        v0.y * v1.z - v0.z * v1.y,
        v0.z * v1.x - v0.x * v1.z,
        v0.x * v1.y - v0.y * v1.x
    };
}

inline Vector3i operator + ( const Vector3i& v0, const Vector3i& v1 )
{
    return{ v0.x + v1.x, v0.y + v1.y, v0.z + v1.z };
}

inline Vector3i operator + ( const Vector3i& v, int i )
{
    return{ v.x + i, v.y + i, v.z + i };
}

inline Vector3i operator + ( int i, const Vector3i& v )
{
    return v + i;
}

inline Vector3f operator + ( const Vector3i& v, float f )
{
    return{ v.x + f, v.y + f, v.z + f };
}

inline Vector3f operator + ( float f, const Vector3i& v )
{
    return{ v + f };
}

inline Vector3i operator - ( const Vector3i& v0, const Vector3i& v1 )
{
    return{ v0.x - v1.x, v0.y - v1.y, v0.z - v1.z };
}

inline Vector3i operator - ( const Vector3i& v, int i )
{
    return{ v.x - i, v.y - i, v.z - i };
}

inline Vector3i operator - ( int i, const Vector3i& v )
{
    return{ i - v.x, i - v.y, i - v.z };
}

inline Vector3f operator - ( const Vector3i& v, float f )
{
    return{ v.x - f, v.y - f, v.z - f };
}

inline Vector3f operator - ( float f, const Vector3i& v )
{
    return{ f - v.x, f - v.y, f - v.z };
}

inline Vector3i operator * ( const Vector3i& v0, const Vector3i& v1 )
{
    return{ v0.x * v1.x, v0.y * v1.y, v0.z * v1.z };
}

inline Vector3i operator - ( const Vector3i& v )
{
    return{ -v.x, -v.y, -v.z };
}

inline Vector3i operator * ( int c, const Vector3i& v )
{
    return{ c * v.x, c * v.y, c * v.z };
}

inline Vector3i operator * ( const Vector3i& v, int c )
{
    return{ c * v.x, c * v.y, c * v.z };
}

inline Vector3f operator * ( float f, const Vector3i& v )
{
    return Vector3f( f * v.x, f * v.y, f * v.z );
}

inline Vector3f operator * ( const Vector3i& v, float f )
{
    return Vector3f( f * v.x, f * v.y, f * v.z );
}

inline Vector3i operator / ( const Vector3i& v0, const Vector3i& v1 )
{
    return{ v0.x / v1.x, v0.y / v1.y, v0.z / v1.z };
}

inline Vector3i operator / ( const Vector3i& v, int i )
{
    return{ v.x / i, v.y / i, v.z / i };
}

inline Vector3f operator / ( const Vector3i& v, float f )
{
    return{ v.x / f, v.y / f, v.z / f };
}

inline Vector3i operator / ( int i, const Vector3i& v )
{
    return{ i / v.x, i / v.y, i / v.z };
}

inline Vector3f operator / ( float f, const Vector3i& v )
{
    return{ f / v.x, f / v.y, f / v.z };
}

inline bool operator == ( const Vector3i& v0, const Vector3i& v1 )
{
    return
    (
        v0.x == v1.x &&
        v0.y == v1.y &&
        v0.z == v1.z
    );
}

inline bool operator != ( const Vector3i& v0, const Vector3i& v1 )
{
    return !( v0 == v1 );
}
