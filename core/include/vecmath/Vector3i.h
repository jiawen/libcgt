#ifndef VECTOR_3I_H
#define VECTOR_3I_H

class Vector2i;
class Vector3f;

class Vector3i
{
public:

	Vector3i();
	Vector3i( int x, int y, int z );
	Vector3i( const Vector2i& xy, int z );
	Vector3i( int x, const Vector2i& yz );

	Vector3i( const Vector3i& rv ); // copy constructor
	Vector3i& operator = ( const Vector3i& rv ); // assignment operator
	// no destructor necessary

	// returns the ith element (mod 3)
	const int& operator [] ( int i ) const;
	int& operator [] ( int i );

	int& x();
	int& y();
	int& z();

	int x() const;
	int y() const;
	int z() const;

	Vector2i xy() const;
	Vector2i xz() const;
	Vector2i yz() const;
	// TODO: all the other combinations

	Vector3i xyz() const;
	Vector3i yzx() const;
	Vector3i zxy() const;
	// TODO: all the other combinations

	float abs() const;
	int absSquared() const;
	Vector3f normalized() const;

	void negate();

	// ---- Utility ----
	operator const int* (); // automatic type conversion for GL
	void print() const;

	static int dot( const Vector3i& v0, const Vector3i& v1 );	

	static Vector3i cross( const Vector3i& v0, const Vector3i& v1 );

	static Vector3f lerp( const Vector3i& v0, const Vector3i& v1, float alpha );

private:

	int m_elements[ 3 ];

};

bool operator == ( const Vector3i& v0, const Vector3i& v1 );
bool operator != ( const Vector3i& v0, const Vector3i& v1 );

Vector3i operator + ( const Vector3i& v0, const Vector3i& v1 );
Vector3i operator - ( const Vector3i& v0, const Vector3i& v1 );
Vector3i operator * ( const Vector3i& v0, const Vector3i& v1 );
Vector3i operator / ( const Vector3i& v0, const Vector3i& v1 );

Vector3i operator - ( const Vector3i& v );
Vector3i operator * ( int c, const Vector3i& v );
Vector3i operator * ( const Vector3i& v, int c );

Vector3f operator * ( float f, const Vector3i& v );
Vector3f operator * ( const Vector3i& v, float f );

Vector3i operator / ( const Vector3i& v, int c );

#endif // VECTOR_3I_H
