#ifndef BOUNDING_BOX_2F
#define BOUNDING_BOX_2F

#include <vecmath/Vector2f.h>

class BoundingBox2f
{
public:

	// constructs an invalid bounding box with
	// min = FLT_MAX, max = FLT_MIN
	// so that merge( this, a ) = a
	BoundingBox2f();

	BoundingBox2f( float minX, float minY,
		float maxX, float maxY );
	BoundingBox2f( const Vector2f& min, const Vector2f& max );
	BoundingBox2f( const BoundingBox2f& rb );
	BoundingBox2f& operator = ( const BoundingBox2f& rb ); // assignment operator
	// no destructor necessary

	void print();

	Vector2f& minimum();
	Vector2f& maximum();

	Vector2f minimum() const;
	Vector2f maximum() const;
	
	Vector2f range() const;
	Vector2f center() const;

	// returns the smallest bounding box that contains both bounding boxes
	static BoundingBox2f merge( const BoundingBox2f& b0, const BoundingBox2f& b1 );

private:

	Vector2f m_min;
	Vector2f m_max;

};

#endif
