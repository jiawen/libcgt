#include "geometry/BoxUtils.h"

void libcgt::core::geometry::boxutils::writeAxisAlignedSolidBox( const Box3f& box,
    Array1DView< Vector4f > vertexPositions )
{
    float x = box.origin().x;
    float y = box.origin().y;
    float z = box.origin().z;
    float width = box.size().x;
    float height = box.size().y;
    float depth = box.size().z;

    // TODO: z may be flipped on all of these, since z = 0 is the back, z = d is the front

	// front
	vertexPositions[  0 ] = Vector4f( x, y, z, 1 );
	vertexPositions[  1 ] = Vector4f( x + width, y, z, 1 );
	vertexPositions[  2 ] = Vector4f( x, y + height, z, 1 );
	vertexPositions[  3 ] = vertexPositions[ 2 ];
	vertexPositions[  4 ] = vertexPositions[ 1 ];
	vertexPositions[  5 ] = Vector4f( x + width, y + height, z, 1 );

	// right
	vertexPositions[  6 ] = Vector4f( x + width, y, z, 1 );
	vertexPositions[  7 ] = Vector4f( x + width, y, z + depth, 1 );
	vertexPositions[  8 ] = Vector4f( x + width, y + height, z, 1 );
	vertexPositions[  9 ] = vertexPositions[ 8 ];
	vertexPositions[ 10 ] = vertexPositions[ 7 ];
	vertexPositions[ 11 ] = Vector4f( x + width, y + height, z + depth, 1 );

	// back
	vertexPositions[ 12 ] = Vector4f( x + width, y, z + depth, 1 );
	vertexPositions[ 13 ] = Vector4f( x, y, z + depth, 1 );
	vertexPositions[ 14 ] = Vector4f( x + width, y + height, z + depth, 1 );
	vertexPositions[ 15 ] = vertexPositions[ 14 ];
	vertexPositions[ 16 ] = vertexPositions[ 13 ];
	vertexPositions[ 17 ] = Vector4f( x, y + height, z + depth, 1 );

	// left
	vertexPositions[ 18 ] = Vector4f( x, y, z + depth, 1 );
	vertexPositions[ 19 ] = Vector4f( x, y, z, 1 );
	vertexPositions[ 20 ] = Vector4f( x, y + height, z + depth, 1 );
	vertexPositions[ 21 ] = vertexPositions[ 20 ];
	vertexPositions[ 22 ] = vertexPositions[ 19 ];
	vertexPositions[ 23 ] = Vector4f( x, y + height, z, 1 );

	// top
	vertexPositions[ 24 ] = Vector4f( x, y + height, z, 1 );
	vertexPositions[ 25 ] = Vector4f( x + width, y + height, z, 1 );
	vertexPositions[ 26 ] = Vector4f( x, y + height, z + depth, 1 );
	vertexPositions[ 27 ] = vertexPositions[ 26 ];
	vertexPositions[ 28 ] = vertexPositions[ 25 ];
	vertexPositions[ 29 ] = Vector4f( x + width, y + height, z + depth, 1 );

	// bottom
	vertexPositions[ 30 ] = Vector4f( x, y, z + depth, 1 );
	vertexPositions[ 31 ] = Vector4f( x + width, y, z + depth, 1 );
	vertexPositions[ 32 ] = Vector4f( x, y, z, 1 );
	vertexPositions[ 33 ] = vertexPositions[ 32 ];
	vertexPositions[ 34 ] = vertexPositions[ 31 ];
	vertexPositions[ 35 ] = Vector4f( x + width, y, z, 1 );
}

void libcgt::core::geometry::boxutils::writeAxisAlignedSolidBoxTextureCoordinates(
    Array1DView< Vector2f > vertexTextureCoordinates )
{
	// front
	vertexTextureCoordinates[  0 ] = Vector2f( 0, 0 );
	vertexTextureCoordinates[  1 ] = Vector2f( 1, 0 );
	vertexTextureCoordinates[  2 ] = Vector2f( 0, 1 );
	vertexTextureCoordinates[  3 ] = vertexTextureCoordinates[ 2 ];
	vertexTextureCoordinates[  4 ] = vertexTextureCoordinates[ 1 ];
	vertexTextureCoordinates[  5 ] = Vector2f( 1, 1 );

	// right
    vertexTextureCoordinates[  6 ] = Vector2f( 0, 0 );
	vertexTextureCoordinates[  7 ] = Vector2f( 1, 0 );
	vertexTextureCoordinates[  8 ] = Vector2f( 0, 1 );
	vertexTextureCoordinates[  9 ] = vertexTextureCoordinates[ 8 ];
	vertexTextureCoordinates[ 10 ] = vertexTextureCoordinates[ 7 ];
    vertexTextureCoordinates[ 11 ] = Vector2f( 1, 1 );

	// back
	vertexTextureCoordinates[ 12 ] = Vector2f( 0, 0 );
	vertexTextureCoordinates[ 13 ] = Vector2f( 1, 0 );
	vertexTextureCoordinates[ 14 ] = Vector2f( 0, 1 );
	vertexTextureCoordinates[ 15 ] = vertexTextureCoordinates[ 14 ];
	vertexTextureCoordinates[ 16 ] = vertexTextureCoordinates[ 13 ];
	vertexTextureCoordinates[ 17 ] = Vector2f( 1, 1 );

	// left
	vertexTextureCoordinates[ 18 ] = Vector2f( 0, 0 );
	vertexTextureCoordinates[ 19 ] = Vector2f( 1, 0 );
	vertexTextureCoordinates[ 20 ] = Vector2f( 0, 1 );
	vertexTextureCoordinates[ 21 ] = vertexTextureCoordinates[ 20 ];
	vertexTextureCoordinates[ 22 ] = vertexTextureCoordinates[ 19 ];
	vertexTextureCoordinates[ 23 ] = Vector2f( 1, 1 );

	// top
	vertexTextureCoordinates[ 24 ] = Vector2f( 0, 0 );
	vertexTextureCoordinates[ 25 ] = Vector2f( 1, 0 );
	vertexTextureCoordinates[ 26 ] = Vector2f( 0, 1 );
	vertexTextureCoordinates[ 27 ] = vertexTextureCoordinates[ 26 ];
	vertexTextureCoordinates[ 28 ] = vertexTextureCoordinates[ 25 ];
	vertexTextureCoordinates[ 29 ] = Vector2f( 1, 1 );

	// bottom
	vertexTextureCoordinates[ 30 ] = Vector2f( 0, 0 );
	vertexTextureCoordinates[ 31 ] = Vector2f( 1, 0 );
	vertexTextureCoordinates[ 32 ] = Vector2f( 0, 1 );
	vertexTextureCoordinates[ 33 ] = vertexTextureCoordinates[ 32 ];
	vertexTextureCoordinates[ 34 ] = vertexTextureCoordinates[ 31 ];
	vertexTextureCoordinates[ 35 ] = Vector2f( 1, 1 );
}

void libcgt::core::geometry::boxutils::writeAxisAlignedWireframeBox( const Box3f& box,
    Array1DView< Vector4f > vertexPositions )
{
    float x = box.origin().x;
    float y = box.origin().y;
    float z = box.origin().z;
    float width = box.size().x;
    float height = box.size().y;
    float depth = box.size().z;

	// front
	vertexPositions[  0 ] = Vector4f( x, y, z, 1 );
	vertexPositions[  1 ] = Vector4f( x + width, y, z, 1 );

	vertexPositions[  2 ] = vertexPositions[  1 ];
	vertexPositions[  3 ] = Vector4f( x + width, y + height, z, 1 );

	vertexPositions[  4 ] = vertexPositions[  3 ];
	vertexPositions[  5 ] = Vector4f( x, y + height, z, 1 );

	// right
	vertexPositions[  6 ] = Vector4f( x + width, y, z, 1 );
	vertexPositions[  7 ] = Vector4f( x + width, y, z + depth, 1 );

	vertexPositions[  8 ] = vertexPositions[  7 ];
	vertexPositions[  9 ] = Vector4f( x + width, y + height, z + depth, 1 );

	vertexPositions[ 10 ] = vertexPositions[  9 ];
	vertexPositions[ 11 ] = Vector4f( x + width, y + height, z, 1 );

	// back
	vertexPositions[ 12 ] = Vector4f( x + width, y, z + depth, 1 );
	vertexPositions[ 13 ] = Vector4f( x, y, z + depth, 1 );

	vertexPositions[ 14 ] = vertexPositions[ 13 ];
	vertexPositions[ 15 ] = Vector4f( x, y + height, z + depth, 1 );

	vertexPositions[ 16 ] = vertexPositions[ 15 ];
	vertexPositions[ 17 ] = Vector4f( x + width, y + height, z + depth, 1 );

	// left
	vertexPositions[ 18 ] = Vector4f( x, y, z + depth, 1 );
	vertexPositions[ 19 ] = Vector4f( x, y, z, 1 );

	vertexPositions[ 20 ] = vertexPositions[ 19 ];
	vertexPositions[ 21 ] = Vector4f( x, y + height, z, 1 );

	vertexPositions[ 22 ] = vertexPositions[ 21 ];
	vertexPositions[ 23 ] = Vector4f( x, y + height, z + depth, 1 );		
}

void libcgt::core::geometry::boxutils::writeAxisAlignedWireframeGrid(
    const Box3f& box, const Vector3i& resolution,
	Array1DView< Vector4f > vertexPositions )
{
	Vector3f delta = box.size() / resolution;
	Vector3f dx( 1, 0, 0 );
	Vector3f dy( 0, 1, 0 );
	Vector3f dz( 0, 0, 1 );

	int k = 0;

	for( int z = 0; z < resolution.z + 1; ++z )
	{
		for( int y = 0; y < resolution.y + 1; ++y )
		{
			vertexPositions[ 2 * k ] = Vector4f( box.origin() + y * delta.y * dy + z * delta.z * dz, 1 );
			vertexPositions[ 2 * k + 1 ] = Vector4f( box.origin() + box.size().x * dx + y * delta.y * dy + z * delta.z * dz, 1 );

			++k;
		}
	}

	for( int z = 0; z < resolution.z + 1; ++z )
	{
		for( int x = 0; x < resolution.x + 1; ++x )
		{
			vertexPositions[ 2 * k ] = Vector4f( box.origin() + x * delta.x * dx + z * delta.z * dz, 1 );
			vertexPositions[ 2 * k + 1 ] = Vector4f( box.origin() + x * delta.x * dx + box.size().y * dy + z * delta.z * dz, 1 );

			++k;
		}
	}

	for( int y = 0; y < resolution.y + 1; ++y )
	{
		for( int x = 0; x < resolution.x + 1; ++x )
		{
			vertexPositions[ 2 * k ] = Vector4f( box.origin() + x * delta.x * dx + y * delta.y * dy, 1 );
			vertexPositions[ 2 * k + 1 ] = Vector4f( box.origin() + x * delta.x * dx + y * delta.y * dy + box.size().z * dz, 1 );

			++k;
		}
	}
}
