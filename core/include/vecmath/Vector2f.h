#pragma once

#include <cmath>

class QString;

class Vector2d;
class Vector2i;
class Vector3f;

class Vector2f
{
public:

	// TODO: conversion operators for double <--> float on Vector3f and Vector4f

    Vector2f(); // (0,0)
    explicit Vector2f( float f ); // (f,f)
    Vector2f( float _x, float _y );

	// copy constructors
    Vector2f( const Vector2f& v );
	Vector2f( const Vector2d& rv );
	Vector2f( const Vector2i& rv );

	// assignment operators
	Vector2f& operator = ( const Vector2f& v );
	Vector2f& operator = ( const Vector2d& v );
	Vector2f& operator = ( const Vector2i& v );

	// no destructor necessary

	// returns the ith element
    const float& operator [] ( int i ) const;
	float& operator [] ( int i );

    Vector2f xy() const;
	Vector2f yx() const;
	Vector2f xx() const;
	Vector2f yy() const;

	// returns ( -y, x )
	Vector2f normal() const;
		
	float norm() const;
	float normSquared() const;

    void normalize();
    Vector2f normalized() const;

    void negate();	

	// ---- Utility ----
    operator const float* () const;
    operator float* ();
	void print() const;
	QString toString() const;

    static float dot( const Vector2f& v0, const Vector2f& v1 );

	// returns (0,0, x0 * y1 - x1 * y0 )
	static Vector3f cross( const Vector2f& v0, const Vector2f& v1 );

	// returns v0 * ( 1 - alpha ) * v1 * alpha
	static Vector2f lerp( const Vector2f& v0, const Vector2f& v1, float alpha );

	Vector2f& operator += ( const Vector2f& v );
	Vector2f& operator -= ( const Vector2f& v );
	Vector2f& operator *= ( float f );
	Vector2f& operator /= ( float f );

	union
	{
		struct
		{
			float x;
			float y;
		};
		float m_elements[ 2 ];
	};
};

Vector2f operator + ( const Vector2f& v0, const Vector2f& v1 );

Vector2f operator - ( const Vector2f& v0, const Vector2f& v1 );
// negate
Vector2f operator - ( const Vector2f& v );

Vector2f operator * ( float f, const Vector2f& v );
Vector2f operator * ( const Vector2f& v, float f );

// component-wise multiplication
Vector2f operator * ( const Vector2f& v0, const Vector2f& v1 );

// component-wise division
Vector2f operator / ( const Vector2f& v, float f );
Vector2f operator / ( const Vector2f& v0, const Vector2f& v1 );

// reciprocal of each component
Vector2f operator / ( float f, const Vector2f& v );

bool operator == ( const Vector2f& v0, const Vector2f& v1 );
bool operator != ( const Vector2f& v0, const Vector2f& v1 );

#include "Vector2f.inl"