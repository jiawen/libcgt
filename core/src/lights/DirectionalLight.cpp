#include "lights/DirectionalLight.h"

#include <geometry/GeometryUtils.h>
#include <vecmath/Vector4f.h>

DirectionalLight::DirectionalLight() :

	m_direction( 0, 0, 1 )

{

}

DirectionalLight::DirectionalLight( const Vector3f& direction ) :

	m_direction( direction )

{

}

Vector3f DirectionalLight::direction() const
{
	return m_direction;
}

void DirectionalLight::setDirection( const Vector3f& direction )
{
	m_direction = direction;
}

// virtual
Matrix3f DirectionalLight::lightBasis() const
{
	return GeometryUtils::getRightHandedBasisWithPreferredUp( m_direction, Vector3f::UP );
}

// virtual
Matrix4f DirectionalLight::lightMatrix( const Camera& camera, const BoundingBox3f& sceneBoundingBox )
{
    const float feather = 1.01f;

	Matrix3f lightLinear = lightBasis();
	Vector3f eye = camera.eye();

	// get the corners of the view frustum in light coordinates
	// with the z = 0 plane at the eye
	std::vector< Vector3f > frustumCorners = camera.frustumCorners();

    BoundingBox3f frustumBB;
    for( int i = 0; i < frustumCorners.size(); ++i )
	{
		// TODO: enlargeToInclude
        frustumBB.enlarge( frustumCorners[i] );
	}

    BoundingBox3f sceneAndFrustum;
	bool intersects = BoundingBox3f::intersect( frustumBB, sceneBoundingBox, sceneAndFrustum );

	// TODO: check for intersection

    std::vector< Vector3f > sceneCorners = sceneBoundingBox.corners();
	std::vector< Vector3f > sceneAndFrustumCorners = sceneAndFrustum.corners();

	for( int i = 0; i < sceneAndFrustumCorners.size(); ++i )
	{
        sceneAndFrustumCorners[ i ] = lightLinear * ( sceneAndFrustumCorners[ i ] - eye );
        sceneCorners[ i ] = lightLinear * ( sceneCorners[ i ] - eye );
	}

    BoundingBox3f inLightCoordinates;
    for(int i = 0; i < sceneAndFrustumCorners.size(); ++i)
        inLightCoordinates.enlarge(sceneAndFrustumCorners[i]);

    Vector3f maxCorner = inLightCoordinates.maximum();
    Vector3f minCorner = inLightCoordinates.minimum();

    Vector3f center = inLightCoordinates.center();
    maxCorner = center + feather * (maxCorner - center);
    minCorner = center + feather * (minCorner - center);

	// add eye point
	for(int j = 0; j < 3; ++j)
	{
		maxCorner[j] = std::max( maxCorner[ j ], 0.0f );
		minCorner[j] = std::min( minCorner[ j ], 0.0f );
	}

	// bound the near plane to the scene
	for( int i = 0; i < sceneCorners.size(); ++i )
	{
		minCorner[2] = std::min( minCorner[2], sceneCorners[ i ][ 2 ] );
	}

	// finally, compute the full light matrix
	Matrix4f lightMatrix;
	lightMatrix.setSubmatrix3x3( 0, 0, lightLinear );
	Vector3f origin = 0.5 * ( minCorner + maxCorner );
	origin[2] = minCorner[2];
	lightMatrix.setCol( 3, Vector4f( -origin, 1.f ) - Vector4f( lightLinear * eye, 0.f ) );
	for(int i = 0; i < 3; ++i)
	{
		lightMatrix.setRow( i, lightMatrix.getRow( i ) * ( ( i == 2 ) ? 1.f : 2.f ) / ( maxCorner[i] - minCorner[i] ) );
	}

	return lightMatrix;
}
