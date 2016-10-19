#include "lights/DirectionalLight.h"

#include <algorithm>

#include <geometry/GeometryUtils.h>
#include <geometry/BoxUtils.h>
#include <vecmath/Box3f.h>
#include <vecmath/Vector4f.h>

using libcgt::core::geometry::corners;

// virtual
Matrix3f DirectionalLight::lightBasis() const
{
    return GeometryUtils::getRightHandedBasisWithPreferredUp( m_direction, Vector3f::UP );
}

// virtual
Matrix4f DirectionalLight::lightMatrix( const Camera& camera, const Box3f& sceneBoundingBox )
{
    const float feather = 1.01f;

    Matrix3f lightLinear = lightBasis();
    Vector3f eye = camera.eye();

    // get the corners of the view frustum in light coordinates
    // with the z = 0 plane at the eye
    std::vector< Vector3f > frustumCorners = camera.frustumCorners();

    Box3f frustumBB;
    for( int i = 0; i < frustumCorners.size(); ++i )
    {
        frustumBB.enlargeToContain( frustumCorners[i] );
    }

    Box3f sceneAndFrustum;
    bool intersects = Box3f::intersect( frustumBB, sceneBoundingBox, sceneAndFrustum );

    // TODO: check for intersection

    Array1D< Vector4f > sceneCorners = corners( sceneBoundingBox );
    Array1D< Vector4f > sceneAndFrustumCorners = corners( sceneAndFrustum );

    for( int i = 0; i < sceneAndFrustumCorners.size(); ++i )
    {
        sceneAndFrustumCorners[ i ].xyz =
            lightLinear * ( sceneAndFrustumCorners[ i ].xyz - eye );
        sceneCorners[ i ].xyz = lightLinear * ( sceneCorners[ i ].xyz - eye );
    }

    Box3f inLightCoordinates;
    for( int i = 0; i < sceneAndFrustumCorners.size(); ++i )
    {
        inLightCoordinates.enlargeToContain( sceneAndFrustumCorners[ i ].xyz );
    }

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
