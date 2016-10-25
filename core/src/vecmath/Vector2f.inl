inline Vector2f::Vector2f( float f ) :
    x( f ),
    y( f )
{

}

inline Vector2f::Vector2f( float _x, float _y ) :
    x( _x ),
    y( _y )
{

}

inline Vector2f::Vector2f( const Vector2d& v ) :
    x( static_cast< float >( v.x ) ),
    y( static_cast< float >( v.y ) )
{

}

inline Vector2f::Vector2f( const Vector2i& v ) :
    x( static_cast< float >( v.x ) ),
    y( static_cast< float >( v.y ) )
{

}

inline Vector2f& Vector2f::operator = ( const Vector2d& v )
{
    x = static_cast< float >( v.x );
    y = static_cast< float >( v.y );

    return *this;
}

inline Vector2f& Vector2f::operator = ( const Vector2i& v )
{
    x = static_cast< float >( v.x );
    y = static_cast< float >( v.y );

    return *this;
}

inline const float& Vector2f::operator [] ( int i ) const
{
    return ( &x )[ i ];
}

inline float& Vector2f::operator [] ( int i )
{
    return ( &x )[ i ];
}

inline Vector2f Vector2f::xy() const
{
    return *this;
}

inline Vector2f Vector2f::yx() const
{
    return{ y, x };
}

inline Vector2f Vector2f::xx() const
{
    return{ x, x };
}

inline Vector2f Vector2f::yy() const
{
    return{ y, y };
}

inline Vector2f Vector2f::normal() const
{
    return{ -y, x };
}

inline float Vector2f::norm() const
{
    return sqrt( normSquared() );
}

inline float Vector2f::normSquared() const
{
    return x * x + y * y;
}

inline void Vector2f::normalize()
{
    float n = norm();
    x /= n;
    y /= n;
}

inline Vector2f Vector2f::normalized() const
{
    float n = norm();
    return{ x / n, y / n };
}

inline Vector2f::operator const float* () const
{
    return &x;
}

inline Vector2f::operator float* ()
{
    return &x;
}

inline std::string Vector2f::toString() const
{
    std::ostringstream sstream;
    sstream << "( " << x << ", " << y << ")";
    return sstream.str();
}

// static
inline float Vector2f::dot( const Vector2f& v0, const Vector2f& v1 )
{
    return v0.x * v1.x + v0.y * v1.y;
}

inline Vector2f& Vector2f::operator += ( const Vector2f& v )
{
    x += v.x;
    y += v.y;

    return *this;
}

inline Vector2f& Vector2f::operator -= ( const Vector2f& v )
{
    x -= v.x;
    y -= v.y;

    return *this;
}

inline Vector2f& Vector2f::operator *= ( float f )
{
    x *= f;
    y *= f;

    return *this;
}

inline Vector2f& Vector2f::operator /= ( float f )
{
    x /= f;
    y /= f;

    return *this;
}

inline Vector2f operator + ( const Vector2f& v0, const Vector2f& v1 )
{
    return{ v0.x + v1.x, v0.y + v1.y };
}

inline Vector2f operator + ( const Vector2f& v, float f )
{
    return{ v.x + f, v.y + f };
}

inline Vector2f operator + ( float f, const Vector2f& v )
{
    return v + f;
}

inline Vector2f operator - ( const Vector2f& v0, const Vector2f& v1 )
{
    return{ v0.x - v1.x, v0.y - v1.y };
}

inline Vector2f operator - ( const Vector2f& v, float f )
{
    return{ v.x - f, v.y - f };
}

inline Vector2f operator - ( float f, const Vector2f& v )
{
    return{ f - v.x, f - v.y };
}

inline Vector2f operator - ( const Vector2f& v )
{
    return{ -v.x, -v.y };
}

inline Vector2f operator * ( const Vector2f& v0, const Vector2f& v1 )
{
    return{ v0.x * v1.x, v0.y * v1.y };
}

inline Vector2f operator * ( float f, const Vector2f& v )
{
    return{ f * v.x, f * v.y };
}

inline Vector2f operator * ( const Vector2f& v, float f )
{
    return{ f * v.x, f * v.y };
}

inline Vector2f operator / ( const Vector2f& v0, const Vector2f& v1 )
{
    return{ v0.x / v1.x, v0.y / v1.y };
}

inline Vector2f operator / ( const Vector2f& v, float f )
{
    return{ v.x / f, v.y / f };
}

inline Vector2f operator / ( float f, const Vector2f& v )
{
    return{ f / v.x, f / v.y };
}

inline bool operator == ( const Vector2f& v0, const Vector2f& v1 )
{
    return( v0.x == v1.x && v0.y == v1.y );
}

inline bool operator != ( const Vector2f& v0, const Vector2f& v1 )
{
    return !( v0 == v1 );
}
