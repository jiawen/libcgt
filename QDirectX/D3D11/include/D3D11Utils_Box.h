#pragma once

#include <common/BasicTypes.h>
#include "common/Iterators.h"
#include <geometry/BoundingBox3f.h>

#include "DynamicVertexBuffer.h"
#include "D3D11Utils.h"

class D3D11Utils_Box
{
public:

	static D3D11_BOX createRange( uint x, uint width );
	static D3D11_BOX createRect( uint x, uint y, uint width, uint height );
	static D3D11_BOX createBox( uint x, uint y, uint z, uint width, uint height, uint depth );

	// writes pBuffer with the contents of box
	// pBuffer needs to have capacity 36
	// TODO: start index...
	template< typename T >
	static void writeAxisAlignedSolidBox( const BoundingBox3f& box, const Vector4f& color, DynamicVertexBuffer* pBuffer );

	// writes vertexArray with the contents of box
	// vertexArray needs to have capacity 36
	// TODO: start index...
	template< typename T >
	static void writeAxisAlignedSolidBox( const Vector3f& origin, const Vector3f& size, T* vertexArray );

	// writes the 36 vertices of a triangle list
	// (3 vertices per triangle * 2 triangles per face * 6 faces)
	// tesselating a 3D box into vertexArray
	// TODO: normals
	template< typename T >
	static void writeAxisAlignedSolidBox( float x, float y, float z, float width, float height, float depth, T* vertexArray );

	// writes pBuffer with the contents of box
	// pBuffer needs to have capacity 24
	// TODO: start vertex
	template< typename T >
	static void writeAxisAlignedWireframeBox( const BoundingBox3f& box, const Vector4f& color, DynamicVertexBuffer* pBuffer );

	template< typename T >
	static void writeAxisAlignedWireframeBox( const Vector3f& origin, const Vector3f& size, T* vertexArray );

	// writes the 24 vertices of a line list
	// (12 edges, 2 vertices each)
	template< typename T >
	static void writeAxisAlignedWireframeBox( float x, float y, float z, float width, float height, float depth, T* vertexArray );
	
	// creates a vertex buffer of lines showing a 3D grid
	//
	// returns a DynamicVertexBuffer< VertexPosition4fColor4f > of size
	// 2*((resolution.y+1)*(resolution.z+1)+(resolution.z+1)*(resolution.x+1)+(resolution.x+1)*(resolution.y+1))
	static std::shared_ptr< DynamicVertexBuffer > createAxisAlignedWireframeGrid( ID3D11Device* pDevice,
		const Vector3f& origin, const Vector3f& size, const Vector3i& resolution,
		const Vector4f& color );

	template< typename T >
	static void writeAxisAlignedWireframeGrid( const Vector3f& origin, const Vector3f& size, const Vector3i& resolution,
		const Vector4f& color, T* vertexArray );
};

// static
template< typename T >
void D3D11Utils_Box::writeAxisAlignedSolidBox( const BoundingBox3f& box, const Vector4f& color, DynamicVertexBuffer* pBuffer )
{
	T* vertexArray = pBuffer->mapForWriteDiscardAs< T >();
	writeAxisAlignedSolidBox( box.minimum(), box.range(), vertexArray );

	for( int i = 0; i < 36; ++i )
	{
		vertexArray[i].color = color;
	}

	pBuffer->unmap();
}

// static
template< typename T >
void D3D11Utils_Box::writeAxisAlignedSolidBox( const Vector3f& origin, const Vector3f& size, T* vertexArray )
{
	writeAxisAlignedSolidBox( origin.x, origin.y, origin.z, size.x, size.y, size.z, vertexArray );
}

// static
template< typename T >
void D3D11Utils_Box::writeAxisAlignedSolidBox( float x, float y, float z, float width, float height, float depth, T* vertexArray )
{
	// front
	vertexArray[  0 ].position = Vector4f( x, y, z, 1 );
	vertexArray[  1 ].position = Vector4f( x + width, y, z, 1 );
	vertexArray[  2 ].position = Vector4f( x, y + height, z, 1 );
	vertexArray[  3 ].position = vertexArray[ 2 ].position;
	vertexArray[  4 ].position = vertexArray[ 1 ].position;
	vertexArray[  5 ].position = Vector4f( x + width, y + height, z, 1 );

	// right
	vertexArray[  6 ].position = Vector4f( x + width, y, z, 1 );
	vertexArray[  7 ].position = Vector4f( x + width, y, z + depth, 1 );
	vertexArray[  8 ].position = Vector4f( x + width, y + height, z, 1 );
	vertexArray[  9 ].position = vertexArray[ 8 ].position;
	vertexArray[ 10 ].position = vertexArray[ 7 ].position;
	vertexArray[ 11 ].position = Vector4f( x + width, y + height, z + depth, 1 );

	// back
	vertexArray[ 12 ].position = Vector4f( x + width, y, z + depth, 1 );
	vertexArray[ 13 ].position = Vector4f( x, y, z + depth, 1 );
	vertexArray[ 14 ].position = Vector4f( x + width, y + height, z + depth, 1 );
	vertexArray[ 15 ].position = vertexArray[ 14 ].position;
	vertexArray[ 16 ].position = vertexArray[ 13 ].position;
	vertexArray[ 17 ].position = Vector4f( x, y + height, z + depth, 1 );

	// left
	vertexArray[ 18 ].position = Vector4f( x, y, z + depth, 1 );
	vertexArray[ 19 ].position = Vector4f( x, y, z, 1 );
	vertexArray[ 20 ].position = Vector4f( x, y + height, z + depth, 1 );
	vertexArray[ 21 ].position = vertexArray[ 20 ].position;
	vertexArray[ 22 ].position = vertexArray[ 19 ].position;
	vertexArray[ 23 ].position = Vector4f( x, y + height, z, 1 );

	// top
	vertexArray[ 24 ].position = Vector4f( x, y + height, z, 1 );
	vertexArray[ 25 ].position = Vector4f( x + width, y + height, z, 1 );
	vertexArray[ 26 ].position = Vector4f( x, y + height, z + depth, 1 );
	vertexArray[ 27 ].position = vertexArray[ 26 ].position;
	vertexArray[ 28 ].position = vertexArray[ 25 ].position;
	vertexArray[ 29 ].position = Vector4f( x + width, y + height, z + depth, 1 );

	// bottom
	vertexArray[ 30 ].position = Vector4f( x, y, z + depth, 1 );
	vertexArray[ 31 ].position = Vector4f( x + width, y, z + depth, 1 );
	vertexArray[ 32 ].position = Vector4f( x, y, z, 1 );
	vertexArray[ 33 ].position = vertexArray[ 32 ].position;
	vertexArray[ 34 ].position = vertexArray[ 31 ].position;
	vertexArray[ 35 ].position = Vector4f( x + width, y, z, 1 );
}

// static
template< typename T >
void D3D11Utils_Box::writeAxisAlignedWireframeBox( const BoundingBox3f& box, const Vector4f& color, DynamicVertexBuffer* pBuffer )
{
	T* vertexArray = pBuffer->mapForWriteDiscardAs< T >();
	writeAxisAlignedWireframeBox( box.minimum(), box.range(), vertexArray );

	for( int i = 0; i < 24; ++i )
	{
		vertexArray[i].color = color;
	}

	pBuffer->unmap();
}

// static
template< typename T >
void D3D11Utils_Box::writeAxisAlignedWireframeBox( const Vector3f& origin, const Vector3f& size, T* vertexArray )
{
	writeAxisAlignedWireframeBox( origin.x, origin.y, origin.z, size.x, size.y, size.z, vertexArray );
}

// static
template< typename T >
void D3D11Utils_Box::writeAxisAlignedWireframeBox( float x, float y, float z, float width, float height, float depth, T* vertexArray )
{
	// front
	vertexArray[  0 ].position = Vector4f( x, y, z, 1 );
	vertexArray[  1 ].position = Vector4f( x + width, y, z, 1 );

	vertexArray[  2 ].position = vertexArray[  1 ].position;
	vertexArray[  3 ].position = Vector4f( x + width, y + height, z, 1 );

	vertexArray[  4 ].position = vertexArray[  3 ].position;
	vertexArray[  5 ].position = Vector4f( x, y + height, z, 1 );

	// right
	vertexArray[  6 ].position = Vector4f( x + width, y, z, 1 );
	vertexArray[  7 ].position = Vector4f( x + width, y, z + depth, 1 );

	vertexArray[  8 ].position = vertexArray[  7 ].position;
	vertexArray[  9 ].position = Vector4f( x + width, y + height, z + depth, 1 );

	vertexArray[ 10 ].position = vertexArray[  9 ].position;
	vertexArray[ 11 ].position = Vector4f( x + width, y + height, z, 1 );

	// back
	vertexArray[ 12 ].position = Vector4f( x + width, y, z + depth, 1 );
	vertexArray[ 13 ].position = Vector4f( x, y, z + depth, 1 );

	vertexArray[ 14 ].position = vertexArray[ 13 ].position;
	vertexArray[ 15 ].position = Vector4f( x, y + height, z + depth, 1 );

	vertexArray[ 16 ].position = vertexArray[ 15 ].position;
	vertexArray[ 17 ].position = Vector4f( x + width, y + height, z + depth, 1 );

	// left
	vertexArray[ 18 ].position = Vector4f( x, y, z + depth, 1 );
	vertexArray[ 19 ].position = Vector4f( x, y, z, 1 );

	vertexArray[ 20 ].position = vertexArray[ 19 ].position;
	vertexArray[ 21 ].position = Vector4f( x, y + height, z, 1 );

	vertexArray[ 22 ].position = vertexArray[ 21 ].position;
	vertexArray[ 23 ].position = Vector4f( x, y + height, z + depth, 1 );		
}

// static
template< typename T >
void D3D11Utils_Box::writeAxisAlignedWireframeGrid( const Vector3f& origin, const Vector3f& size, const Vector3i& resolution,
	const Vector4f& color, T* pVertexArray )
{
	Vector3f delta = size / resolution;
	Vector3f dx( 1, 0, 0 );
	Vector3f dy( 0, 1, 0 );
	Vector3f dz( 0, 0, 1 );

	int k = 0;

	for( int z = 0; z < resolution.z + 1; ++z )
	{
		for( int y = 0; y < resolution.y + 1; ++y )
		{
			pVertexArray[ 2 * k ].position = Vector4f( origin + y * delta.y * dy + z * delta.z * dz, 1 );
			pVertexArray[ 2 * k ].color = color;

			pVertexArray[ 2 * k + 1 ].position = Vector4f( origin + size.x * dx + y * delta.y * dy + z * delta.z * dz, 1 );
			pVertexArray[ 2 * k + 1 ].color = color;

			++k;
		}
	}

	for( int z = 0; z < resolution.z + 1; ++z )
	{
		for( int x = 0; x < resolution.x + 1; ++x )
		{
			pVertexArray[ 2 * k ].position = Vector4f( origin + x * delta.x * dx + z * delta.z * dz, 1 );
			pVertexArray[ 2 * k ].color = color;

			pVertexArray[ 2 * k + 1 ].position = Vector4f( origin + x * delta.x * dx + size.y * dy + z * delta.z * dz, 1 );
			pVertexArray[ 2 * k + 1 ].color = color;

			++k;
		}
	}

	for( int y = 0; y < resolution.y + 1; ++y )
	{
		for( int x = 0; x < resolution.x + 1; ++x )
		{
			pVertexArray[ 2 * k ].position = Vector4f( origin + x * delta.x * dx + y * delta.y * dy, 1 );
			pVertexArray[ 2 * k ].color = color;

			pVertexArray[ 2 * k + 1 ].position = Vector4f( origin + x * delta.x * dx + y * delta.y * dy + size.z * dz, 1 );
			pVertexArray[ 2 * k + 1 ].color = color;

			++k;
		}
	}
}
