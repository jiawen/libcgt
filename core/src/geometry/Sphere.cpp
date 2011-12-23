#include "geometry/Sphere.h"

#include <cmath>

#include "math/MathUtils.h"

Sphere::Sphere( float radius, const Vector3f& center ) :

	m_radius( radius ),
	m_center( center )

{

}

std::vector< Vector3f > Sphere::tesselate( int nTheta, int nPhi )
{
	std::vector< Vector3f > vertices;
	vertices.reserve( 6 * nTheta * nPhi );

	float dt = MathUtils::TWO_PI / nTheta;
	float dp = MathUtils::PI / nPhi;

	for( int t = 0; t < nTheta; ++t )
	{
		float t0 = t * dt;
		float t1 = t0 + dt;

		for( int p = 0; p < nPhi; ++p )
		{
			float p0 = p * dp;
			float p1 = p0 + dp;

			Vector3f v00 = m_center +
				Vector3f
				(
					m_radius * cos( t0 ) * sin( p0 ),
					m_radius * sin( t0 ) * sin( p0 ),
					m_radius * cos( p0 )
				);

			Vector3f v10 = m_center +
				Vector3f
				(
					m_radius * cos( t1 ) * sin( p0 ),
					m_radius * sin( t1 ) * sin( p0 ),
					m_radius * cos( p0 )
				);

			Vector3f v01 = m_center +
				Vector3f
				(
					m_radius * cos( t0 ) * sin( p1 ),
					m_radius * sin( t0 ) * sin( p1 ),
					m_radius * cos( p1 )
				);

			Vector3f v11 = m_center +
				Vector3f
				(
					m_radius * cos( t1 ) * sin( p1 ),
					m_radius * sin( t1 ) * sin( p1 ),
					m_radius * cos( p1 )
				);

			vertices.push_back( v00 );
			vertices.push_back( v10 );
			vertices.push_back( v01 );

			vertices.push_back( v01 );
			vertices.push_back( v10 );
			vertices.push_back( v11 );
		}
	}

	return vertices;
}