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

/*
inline Vector2f::Vector2f( std::initializer_list< float > xy )
{
    x = *( xy.begin() );
    y = *( xy.begin() + 1 );
}*/

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
    return Vector2f{ y, x };
}

inline Vector2f Vector2f::xx() const
{
    return Vector2f{ x, x };
}

inline Vector2f Vector2f::yy() const
{
    return Vector2f{ y, y };
}

inline Vector2f Vector2f::normal() const
{
    return Vector2f{ -y, x };
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

inline void Vector2f::negate()
{
    x = -x;
    y = -y;
}

inline Vector2f::operator const float* () const
{
    return &x;
}

inline Vector2f::operator float* ()
{
    return &x;
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
    return Vector2f{ v0.x + v1.x, v0.y + v1.y };
}

inline Vector2f operator - ( const Vector2f& v0, const Vector2f& v1 )
{
    return Vector2f{ v0.x - v1.x, v0.y - v1.y };
}

inline Vector2f operator - ( const Vector2f& v )
{
    return Vector2f{ -v.x, -v.y };
}

inline Vector2f operator * ( const Vector2f& v0, const Vector2f& v1 )
{
    return Vector2f{ v0.x * v1.x, v0.y * v1.y };
}

inline Vector2f operator * ( float f, const Vector2f& v )
{
    return Vector2f{ f * v.x, f * v.y };
}

inline Vector2f operator * ( const Vector2f& v, float f )
{
    return Vector2f{ f * v.x, f * v.y };
}

inline Vector2f operator / ( const Vector2f& v0, const Vector2f& v1 )
{
    return Vector2f{ v0.x / v1.x, v0.y / v1.y };
}

inline Vector2f operator / ( const Vector2f& v, float f )
{
    return Vector2f{ v.x / f, v.y / f };
}

inline Vector2f operator / ( float f, const Vector2f& v )
{
    return Vector2f{ f / v.x, f / v.y };
}

inline bool operator == ( const Vector2f& v0, const Vector2f& v1 )
{
    return( v0.x == v1.x && v0.y == v1.y );
}

inline bool operator != ( const Vector2f& v0, const Vector2f& v1 )
{
    return !( v0 == v1 );
}
