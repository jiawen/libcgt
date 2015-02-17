inline Vector4f::Vector4f()
{
    x = 0;
    y = 0;
    z = 0;
    w = 0;
}

inline Vector4f::Vector4f( float f )
{
    x = f;
    y = f;
    z = f;
    w = f;
}

inline Vector4f::Vector4f( float _x, float _y, float _z, float _w )
{
    x = _x;
    y = _y;
    z = _z;
    w = _w;
}

inline const float& Vector4f::operator [] ( int i ) const
{
	return ( &x )[ i ];
}

inline float& Vector4f::operator [] ( int i )
{
	return ( &x )[ i ];
}

inline Vector4f::operator const float* () const
{
	return &x;
}

inline Vector4f::operator float* ()
{
	return &x;
}

inline Vector4f& Vector4f::operator += ( const Vector4f& v )
{
	x += v.x;
	y += v.y;
	z += v.z;
	w += v.w;

	return *this;
}

inline Vector4f& Vector4f::operator -= ( const Vector4f& v )
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	w -= v.w;

	return *this;
}

inline Vector4f& Vector4f::operator *= ( float f )
{
	x *= f;
	y *= f;
	z *= f;
	w *= f;

	return *this;
}

inline Vector4f& Vector4f::operator /= ( float f )
{
	x /= f;
	y /= f;
	z /= f;
	w /= f;

	return *this;
}

inline Vector4f operator + ( const Vector4f& v0, const Vector4f& v1 )
{
	return Vector4f( v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w );
}

inline Vector4f operator - ( const Vector4f& v0, const Vector4f& v1 )
{
	return Vector4f( v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w );
}

inline Vector4f operator - ( const Vector4f& v )
{
	return Vector4f( -v.x, -v.y, -v.z , -v.w );
}

inline Vector4f operator * ( float f, const Vector4f& v )
{
	return Vector4f( v.x * f, v.y * f, v.z * f, v.w * f );
}

inline Vector4f operator * ( const Vector4f& v, float f )
{
	return Vector4f( v.x * f, v.y * f, v.z * f, v.w * f );
}

inline Vector4f operator * ( const Vector4f& v0, const Vector4f& v1 )
{
	return Vector4f( v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w );
}

inline Vector4f operator / ( const Vector4f& v0, const Vector4f& v1 )
{
	return Vector4f( v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w );
}

inline Vector4f operator / ( const Vector4f& v, float f )
{
	return Vector4f( v.x / f, v.y / f, v.z / f, v.w / f );
}

inline Vector4f operator / ( float f, const Vector4f& v )
{
	return Vector4f( f / v.x, f / v.y, f / v.z, f / v.w );
}

inline bool operator == ( const Vector4f& v0, const Vector4f& v1 )
{
	return( v0.x == v1.x && v0.y == v1.y && v0.z == v1.z && v0.w == v1.w );
}

inline bool operator != ( const Vector4f& v0, const Vector4f& v1 )
{
	return !( v0 == v1 );
}