inline Vector2i::Vector2i( int i ) :
    x( i ),
    y( i )
{

}

inline Vector2i::Vector2i( int _x, int _y ) :
    x( _x ),
    y( _y )
{

}

inline const int& Vector2i::operator [] ( int i ) const
{
    return ( &x )[ i ];
}

inline int& Vector2i::operator [] ( int i )
{
    return ( &x )[ i ];
}

inline Vector2i Vector2i::xy() const
{
    return{ x, y };
}

inline Vector2i Vector2i::yx() const
{
    return{ y, x };
}

inline Vector2i Vector2i::xx() const
{
    return{ x, x };
}

inline Vector2i Vector2i::yy() const
{
    return{ y, y };
}

inline float Vector2i::norm() const
{
    return sqrt( static_cast< float >( normSquared() ) );
}

inline int Vector2i::normSquared() const
{
    return( x * x + y * y );
}

inline Vector2f Vector2i::normalized() const
{
    float n = 1.f / norm();

    return
    {
        n * x,
        n * y
    };
}

inline Vector2i::operator const int* ( ) const
{
    return &x;
}

inline Vector2i::operator int* ( )
{
    return &x;
}

inline std::string Vector2i::toString() const
{
    std::ostringstream sstream;
    sstream << "( " << x << ", " << y << ")";
    return sstream.str();
}

// static
inline int Vector2i::dot( const Vector2i& v0, const Vector2i& v1 )
{
    return v0.x * v1.x + v0.y * v1.y;
}

inline Vector2i& Vector2i::operator += ( const Vector2i& v )
{
    x += v.x;
    y += v.y;

    return *this;
}

inline Vector2i& Vector2i::operator -= ( const Vector2i& v )
{
    x -= v.x;
    y -= v.y;

    return *this;
}

inline Vector2i& Vector2i::operator *= ( int s )
{
    x *= s;
    y *= s;

    return *this;
}

inline Vector2i& Vector2i::operator /= ( int s )
{
    x /= s;
    y /= s;

    return *this;
}

inline Vector2i operator + ( const Vector2i& v0, const Vector2i& v1 )
{
    return{ v0.x + v1.x, v0.y + v1.y };
}

inline Vector2i operator + ( const Vector2i& v, int i )
{
    return{ v.x + i, v.y + i };
}

inline Vector2i operator + ( int i, const Vector2i& v )
{
    return v + i;
}

inline Vector2f operator + ( const Vector2i& v, float f )
{
    return{ v.x + f, v.y + f };
}

inline Vector2f operator + ( float f, const Vector2i& v )
{
    return v + f;
}

inline Vector2i operator - ( const Vector2i& v0, const Vector2i& v1 )
{
    return{ v0.x - v1.x, v0.y - v1.y };
}

inline Vector2i operator - ( const Vector2i& v, int i )
{
    return{ v.x - i, v.y - i };
}

inline Vector2i operator - ( int i, const Vector2i& v )
{
    return{ i - v.x, i - v.y };
}

inline Vector2f operator - ( const Vector2i& v, float f )
{
    return{ v.x - f, v.y - f };
}

inline Vector2f operator - ( float f, const Vector2i& v )
{
    return{ f - v.x, f - v.y };
}

inline Vector2i operator - ( const Vector2i& v )
{
    return{ -v.x, -v.y };
}

inline Vector2i operator * ( int c, const Vector2i& v )
{
    return{ c * v.x, c * v.y };
}

inline Vector2i operator * ( const Vector2i& v, int c )
{
    return{ c * v.x, c * v.y };
}

inline Vector2f operator * ( float f, const Vector2i& v )
{
    return{ f * v.x, f * v.y };
}

inline Vector2f operator * ( const Vector2i& v, float f )
{
    return{ f * v.x, f * v.y };
}

inline Vector2i operator * ( const Vector2i& v0, const Vector2i& v1 )
{
    return{ v0.x * v1.x, v0.y * v1.y };
}

inline Vector2i operator / ( const Vector2i& v0, const Vector2i& v1 )
{
    return{ v0.x / v1.x, v0.y / v1.y };
}

inline Vector2i operator / ( const Vector2i& v, int i )
{
    return{ v.x / i, v.y / i };
}

inline Vector2f operator / ( const Vector2i& v, float f )
{
    return{ v.x / f, v.y / f };
}

inline bool operator == ( const Vector2i& v0, const Vector2i& v1 )
{
    return
    (
        v0.x == v1.x &&
        v0.y == v1.y
    );
}

inline bool operator != ( const Vector2i& v0, const Vector2i& v1 )
{
    return !( v0 == v1 );
}
