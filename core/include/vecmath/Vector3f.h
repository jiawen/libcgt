#pragma once

class QString;

class Vector2f;
class Vector3d;
class Vector3i;

class Vector3f
{
public:

	static const Vector3f ZERO;
	static const Vector3f UP;
	static const Vector3f RIGHT;
	static const Vector3f FORWARD;
	
    Vector3f(); // (0,0,0)
    Vector3f( float f ); // (f,f,f)
    Vector3f( float _x, float _y, float _z );

	Vector3f( const Vector2f& xy, float z );
	Vector3f( float x, const Vector2f& yz );

	// copy constructors
    Vector3f( const Vector3f& v );
	Vector3f( const Vector3d& rv );
	Vector3f( const Vector3i& rv );

	// assignment operators
    Vector3f& operator = ( const Vector3f& v );
	Vector3f& operator = ( const Vector3d& rv );
	Vector3f& operator = ( const Vector3i& rv );

	// no destructor necessary

	// returns the ith element
    const float& operator [] ( int i ) const;
    float& operator [] ( int i );
	
	Vector2f xy() const;
	Vector2f xz() const;
	Vector2f yz() const;
	// TODO: all the other combinations

	Vector3f xyz() const;
	Vector3f yzx() const;
	Vector3f zxy() const;
	// TODO: all the other combinations

	// TODO: these are deprecated, use norm() and normSquared()
	float abs() const { return norm(); }
	float absSquared() const { return normSquared(); }

	float norm() const;
	float normSquared() const;

	void normalize();
	Vector3f normalized() const;

	void homogenize();
	Vector3f homogenized() const;

	void negate();
	
	Vector3i floored() const;

	// automatic type conversion to float pointer
    operator const float* () const;
    operator float* ();
	QString toString() const;

	// dot product
    static float dot( const Vector3f& v0, const Vector3f& v1 );

	// cross product
	static Vector3f cross( const Vector3f& v0, const Vector3f& v1 );

	// returns v0 * ( 1 - alpha ) * v1 * alpha
	static Vector3f lerp( const Vector3f& v0, const Vector3f& v1, float alpha );

	// Catmull-Rom interpolation
	static Vector3f cubicInterpolate( const Vector3f& p0, const Vector3f& p1, const Vector3f& p2, const Vector3f& p3, float t );

	Vector3f& operator += ( const Vector3f& v );
	Vector3f& operator -= ( const Vector3f& v );
    Vector3f& operator *= ( float f );
	Vector3f& operator /= ( float f );

	union
	{
		struct
		{
			float x;
			float y;
			float z;
		};
		float m_elements[3];
	};

};

Vector3f operator + ( const Vector3f& v0, const Vector3f& v1 );

Vector3f operator - ( const Vector3f& v0, const Vector3f& v1 );
// negate
Vector3f operator - ( const Vector3f& v );

Vector3f operator * ( float f, const Vector3f& v );
Vector3f operator * ( const Vector3f& v, float f );

// component-wise multiplication
Vector3f operator * ( const Vector3f& v0, const Vector3f& v1 );

Vector3f operator / ( const Vector3f& v, float f );
// component-wise division
Vector3f operator / ( const Vector3f& v0, const Vector3f& v1 );

#include "Vector3f.inl"
