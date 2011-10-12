#ifndef VECTOR_2I_H
#define VECTOR_2I_H

class Vector2f;
class Vector3i;

class Vector2i
{
public:

	// TODO: conversion operators for double <--> int on Vector3f and Vector4f

	Vector2i();
	Vector2i( int x, int y );
	Vector2i( const Vector2i& rv ); // copy constructor
	Vector2i& operator = ( const Vector2i& rv ); // assignment operator
	// no destructor necessary

	// returns the ith element (mod 2)
	const int& operator [] ( int i ) const;
	int& operator [] ( int i );

	int& x();
	int& y();

	int x() const;
	int y() const;
	Vector2i xy() const;
	Vector2i yx() const;
	Vector2i xx() const;
	Vector2i yy() const;

	float abs() const;
	int absSquared() const;
	Vector2f normalized() const;

	void negate();

	// ---- Utility ----
	operator const int* (); // automatic type conversion for GL
	void print() const;

	static int dot( const Vector2i& v0, const Vector2i& v1 );	

	static Vector3i cross( const Vector2i& v0, const Vector2i& v1 );

	// returns v0 * ( 1 - alpha ) * v1 * alpha
	static Vector2f lerp( const Vector2i& v0, const Vector2i& v1, float alpha );

private:

	int m_elements[2];

};

bool operator == ( const Vector2i& v0, const Vector2i& v1 );

Vector2i operator + ( const Vector2i& v0, const Vector2i& v1 );
Vector2i operator - ( const Vector2i& v0, const Vector2i& v1 );
Vector2i operator * ( const Vector2i& v0, const Vector2i& v1 );
Vector2i operator / ( const Vector2i& v0, const Vector2i& v1 );

Vector2i operator - ( const Vector2i& v );
Vector2i operator * ( int c, const Vector2i& v );
Vector2i operator * ( const Vector2i& v, int c );

Vector2f operator * ( float f, const Vector2i& v );
Vector2f operator * ( const Vector2i& v, float f );

Vector2i operator / ( const Vector2i& v, int c );

#endif // VECTOR_2I_H
