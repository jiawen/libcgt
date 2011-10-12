#ifndef VECTOR_4I_H
#define VECTOR_4I_H

class Vector2i;
class Vector3i;

#include "Vector4f.h"

class Vector4i
{
public:	

	Vector4i();
	Vector4i( int i ); // fills all 4 components with i
	Vector4i( int x, int y, int z, int w );
	Vector4i( const Vector2i& xy, int z, int w );
	Vector4i( int x, const Vector2i& yz, int w );
	Vector4i( int x, int y, const Vector2i& zw );
	Vector4i( const Vector2i& xy, const Vector2i& zw );
	Vector4i( const Vector3i& xyz, int w );
	Vector4i( int x, const Vector3i& yzw );

	Vector4i( const Vector4i& rv ); // copy constructor	
	Vector4i& operator = ( const Vector4i& rv ); // assignment operator
	// no destructor necessary

	// returns the ith element (mod 4)
	const int& operator [] ( int i ) const;
	int& operator [] ( int i );

	int& x();
	int& y();
	int& z();
	int& w();

	int x() const;
	int y() const;
	int z() const;
	int w() const;

	Vector2i xy() const;
	Vector2i yz() const;
	Vector2i zw() const;
	Vector2i wx() const;
	// TODO: the other combinations

	Vector3i xyz() const;
	Vector3i yzw() const;
	Vector3i zwx() const;
	Vector3i wxy() const;

	Vector3i xyw() const;
	Vector3i yzx() const;
	Vector3i zwy() const;
	Vector3i wxz() const;
	// TODO: the rest of the vec3 combinations

	// TODO: swizzle all the vec4s

	float abs() const;
	int absSquared() const;
	Vector4f normalized() const;

	// if v.z != 0, v = v / v.w
	void homogenize();
	Vector4i homogenized() const;

	void negate();

	// ---- Utility ----
	operator const int* (); // automatic type conversion for GL
	void print() const;

	static int dot( const Vector4i& v0, const Vector4i& v1 );
	static Vector4f lerp( const Vector4i& v0, const Vector4i& v1, float alpha );

private:

	int m_elements[ 4 ];

};

Vector4i operator + ( const Vector4i& v0, const Vector4i& v1 );
Vector4i operator - ( const Vector4i& v0, const Vector4i& v1 );
Vector4i operator * ( const Vector4i& v0, const Vector4i& v1 );
Vector4i operator / ( const Vector4i& v0, const Vector4i& v1 );

Vector4i operator - ( const Vector4i& v );
Vector4i operator * ( int c, const Vector4i& v );
Vector4i operator * ( const Vector4i& v, int c );

Vector4f operator * ( float f, const Vector4i& v );
Vector4f operator * ( const Vector4i& v, float f );

Vector4i operator / ( const Vector4i& v, int c );

#endif // VECTOR_4I_H
