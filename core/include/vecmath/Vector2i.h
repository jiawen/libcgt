#pragma once

class QString;

class Vector2f;
class Vector3i;

class Vector2i
{
public:

	// TODO: conversion operators for double <--> int on Vector3f and Vector4f

	Vector2i();
	explicit Vector2i( int i ); // fills both elements with i
	Vector2i( int x, int y );
	Vector2i( const Vector2i& rv ); // copy constructor
	Vector2i& operator = ( const Vector2i& rv ); // assignment operator
	// no destructor necessary

	// returns the ith element
	const int& operator [] ( int i ) const;
	int& operator [] ( int i );

	Vector2i xy() const;
	Vector2i yx() const;
	Vector2i xx() const;
	Vector2i yy() const;

	float norm() const;
	int normSquared() const;
	Vector2f normalized() const;

	void negate();

	Vector2i flippedUD( int height ) const;

	// implicit cast
	operator const int* () const;
	operator int* ();
	QString toString() const;

	static int dot( const Vector2i& v0, const Vector2i& v1 );	

	static Vector3i cross( const Vector2i& v0, const Vector2i& v1 );

	inline Vector2i& operator += ( const Vector2i& v );
	inline Vector2i& operator -= ( const Vector2i& v );
    inline Vector2i& operator *= ( int s );
	inline Vector2i& operator /= ( int s );

	union
	{
		struct
		{
			int x;
			int y;
		};
		int m_elements[2];
	};

};

Vector2i operator + ( const Vector2i& v0, const Vector2i& v1 );

Vector2i operator - ( const Vector2i& v0, const Vector2i& v1 );
// negate
Vector2i operator - ( const Vector2i& v );

Vector2i operator * ( int c, const Vector2i& v );
Vector2i operator * ( const Vector2i& v, int c );
Vector2f operator * ( float f, const Vector2i& v );
Vector2f operator * ( const Vector2i& v, float f );

// component-wise multiplication
Vector2i operator * ( const Vector2i& v0, const Vector2i& v1 );

// component-wise division
Vector2i operator / ( const Vector2i& v0, const Vector2i& v1 );
Vector2i operator / ( const Vector2i& v, int c );

bool operator == ( const Vector2i& v0, const Vector2i& v1 );
bool operator != ( const Vector2i& v0, const Vector2i& v1 );

inline Vector2i& Vector2i::operator += ( const Vector2i& v )
{
	m_elements[ 0 ] += v.m_elements[ 0 ];
	m_elements[ 1 ] += v.m_elements[ 1 ];

	return *this;
}

inline Vector2i& Vector2i::operator -= ( const Vector2i& v )
{
	m_elements[ 0 ] -= v.m_elements[ 0 ];
	m_elements[ 1 ] -= v.m_elements[ 1 ];

	return *this;
}

inline Vector2i& Vector2i::operator *= ( int s )
{
	m_elements[ 0 ] *= s;
	m_elements[ 1 ] *= s;

	return *this;
}

inline Vector2i& Vector2i::operator /= ( int s )
{
	m_elements[ 0 ] /= s;
	m_elements[ 1 ] /= s;

	return *this;
}
