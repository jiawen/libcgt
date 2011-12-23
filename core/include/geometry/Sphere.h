#ifndef SPHERE_H
#define SPHERE_H

#include <vector>

#include "vecmath/Vector3f.h"

class Sphere
{
public:

	Sphere( float radius = 1, const Vector3f& center = Vector3f( 0, 0, 0 ) );

	std::vector< Vector3f > tesselate( int nTheta, int nPhi );

	Vector3f m_center;
	float m_radius;

};

#endif // SPHERE_H
