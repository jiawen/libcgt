#ifndef CYLINDER_H
#define CYLINDER_H

#include <vector>

#include "vecmath/Vector2f.h"
#include "vecmath/Vector3f.h"
#include "vecmath/Vector4f.h"

class Cylinder
{
public:

	Cylinder( float _radius = 1, float _height = 1, const Vector3f& _baseCenter = Vector3f( 0, 0, 0 ) );

	void tesselate( int nTheta, int nHeight,
		std::vector< Vector4f >& positions,
		std::vector< Vector3f >& normals );

	float radius;
	float height;
	Vector3f baseCenter;

};

#endif // CYLINDER_H
