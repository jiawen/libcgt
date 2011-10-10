#ifndef VECTOR_4F_H
#define VECTOR_4F_H

class Vector2f;
class Vector3f;

class Vector4i;
class Vector4d;

class Vector4f
{
public:

	Vector4f();
	Vector4f( float f );
	Vector4f( float fx, float fy, float fz, float fw );
	Vector4f( float buffer[ 4 ] );

	Vector4f( const Vector2f& xy, float z, float w );
	Vector4f( float x, const Vector2f& yz, float w );
	Vector4f( float x, float y, const Vector2f& zw );
	Vector4f( const Vector2f& xy, const Vector2f& zw );

	Vector4f( const Vector3f& xyz, float w );
	Vector4f( float x, const Vector3f& yzw );

	// copy constructors
	Vector4f( const Vector4f& rv );
	Vector4f( const Vector4d& rv );
	Vector4f( const Vector4i& rv );

	// assignment operators
	Vector4f& operator = ( const Vector4f& rv );
	Vector4f& operator = ( const Vector4d& rv );
	Vector4f& operator = ( const Vector4i& rv );

	// no destructor necessary

	operator float* (); // implicit cast

	// returns the ith element (mod 4)
	const float& operator [] ( int i ) const;
	float& operator [] ( int i );

	float& x();
	float& y();
	float& z();
	float& w();

	float x() const;
	float y() const;
	float z() const;
	float w() const;

	Vector2f xy() const;
	Vector2f yz() const;
	Vector2f zw() const;
	Vector2f wx() const;
	// TODO: the other combinations

	Vector3f xyz() const;
	Vector3f yzw() const;
	Vector3f zwx() const;
	Vector3f wxy() const;

	Vector3f xyw() const;
	Vector3f yzx() const;
	Vector3f zwy() const;
	Vector3f wxz() const;
	// TODO: the rest of the vec3 combinations

	// TODO: swizzle all the vec4s

	float abs() const;
	float absSquared() const;
	void normalize();
	Vector4f normalized() const;

	// if v.z != 0, v = v / v.w
	void homogenize();
	Vector4f homogenized() const;

	void negate();

	// ---- Utility ----
	// TODO: make the rest const compliant
	operator const float* () const; // automatic type conversion for GL
	void print() const;

	static float dot( const Vector4f& v0, const Vector4f& v1 );
	static Vector4f lerp( const Vector4f& v0, const Vector4f& v1, float alpha );

private:

	float m_elements[ 4 ];

};

Vector4f operator + ( const Vector4f& v0, const Vector4f& v1 );
Vector4f operator - ( const Vector4f& v0, const Vector4f& v1 );
Vector4f operator * ( const Vector4f& v0, const Vector4f& v1 );
Vector4f operator / ( const Vector4f& v0, const Vector4f& v1 );

Vector4f operator - ( const Vector4f& v );
Vector4f operator * ( float f, const Vector4f& v );
Vector4f operator * ( const Vector4f& v, float f );

#endif // VECTOR_4F_H
