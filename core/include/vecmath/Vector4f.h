#pragma once

#include "Vector2f.h"
#include "Vector3f.h"

class QString;

class Vector4i;
class Vector4d;

class Vector4f
{
public:
	
	Vector4f(); // initialized to 0
	explicit Vector4f( float f ); // initialized to (f, f, f, f )
	Vector4f( float _x, float _y, float _z, float _w );
	Vector4f( float buffer[ 4 ] );

	Vector4f( const Vector2f& _xy, float _z, float _w );
	Vector4f( float _x, const Vector2f& _yz, float _w );
	Vector4f( float _x, float _y, const Vector2f& _zw );
	Vector4f( const Vector2f& _xy, const Vector2f& _zw );

	Vector4f( const Vector3f& _xyz, float _w );
	Vector4f( float _x, const Vector3f& _yzw );

	// copy constructors
	Vector4f( const Vector4f& v );
	Vector4f( const Vector4d& v );
	Vector4f( const Vector4i& v );

	// assignment operators
	Vector4f& operator = ( const Vector4f& v );
	Vector4f& operator = ( const Vector4d& v );
	Vector4f& operator = ( const Vector4i& v );

	// no destructor necessary	

	// returns the ith element
	const float& operator [] ( int i ) const;
	float& operator [] ( int i );
	
	Vector2f yz() const;
	Vector2f zw() const;
	Vector2f wx() const;
	// TODO: the other combinations

	Vector3f yzw() const;
	Vector3f zwx() const;
	Vector3f wxy() const;

	Vector3f xyw() const;
	Vector3f yzx() const;
	Vector3f zwy() const;
	Vector3f wxz() const;
	// TODO: the rest of the vec3 combinations

	// TODO: swizzle all the vec4s

	float norm() const;
	float normSquared() const;

	void normalize();
	Vector4f normalized() const;

	// If v.w != 0, v = v / v.w.
    // Else, does nothing.
	void homogenize();
	Vector4f homogenized() const;

	void negate();

	// implicit cast
	operator const float* () const;
	operator float* ();
	QString toString() const;

	static float dot( const Vector4f& v0, const Vector4f& v1 );

	Vector4f& operator += ( const Vector4f& v );
	Vector4f& operator -= ( const Vector4f& v );
    Vector4f& operator *= ( float f );
	Vector4f& operator /= ( float f );

	union
	{
		struct
		{
			float x;
			float y;
			float z;
			float w;
		};
        struct
        {
            Vector2f xy;
        };
        struct
        {
            Vector3f xyz;            
        };
		float m_elements[4];
	};
};

Vector4f operator + ( const Vector4f& v0, const Vector4f& v1 );

Vector4f operator - ( const Vector4f& v0, const Vector4f& v1 );
// negate
Vector4f operator - ( const Vector4f& v );

Vector4f operator * ( float f, const Vector4f& v );
Vector4f operator * ( const Vector4f& v, float f );

// component-wise multiplication
Vector4f operator * ( const Vector4f& v0, const Vector4f& v1 );

// component-wise division
Vector4f operator / ( const Vector4f& v0, const Vector4f& v1 );
Vector4f operator / ( const Vector4f& v, float f );

// reciprocal of each component
Vector4f operator / ( float f, const Vector4f& v );

inline bool operator == ( const Vector4f& v0, const Vector4f& v1 );
inline bool operator != ( const Vector4f& v0, const Vector4f& v1 );

#include "Vector4f.inl"