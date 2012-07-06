#include "D3D11Utils_Box.h"

// static
D3D11_BOX D3D11Utils_Box::createRange( uint x, uint width )
{
	return createRect( x, 0, width, 1 );
}

// static
D3D11_BOX D3D11Utils_Box::createRect( uint x, uint y, uint width, uint height )
{
	return createBox( x, y, 0, width, height, 1 );
}

// static
D3D11_BOX D3D11Utils_Box::createBox( uint x, uint y, uint z, uint width, uint height, uint depth )
{
	// D3D11_BOX is a half-open interval
	D3D11_BOX box;

	box.left = x;
	box.right = x + width;
	box.top = y;
	box.bottom = y + height;
	box.front = z;
	box.back = z + depth;

	return box;
}

// static
std::shared_ptr< DynamicVertexBuffer > D3D11Utils_Box::createAxisAlignedWireframeGrid( ID3D11Device* pDevice,
	const Vector3f& origin, const Vector3f& size, const Vector3i& resolution,
	const Vector4f& color )
{
	int nVertices = 2 *
	(
		( resolution.y + 1 ) * ( resolution.z + 1 ) +
		( resolution.z + 1 ) * ( resolution.x + 1 ) +
		( resolution.x + 1 ) * ( resolution.y + 1 )
	);

	std::shared_ptr< DynamicVertexBuffer > pBuffer
	(
		new DynamicVertexBuffer( pDevice, nVertices, VertexPosition4fColor4f::sizeInBytes() )
	);

	VertexPosition4fColor4f* pArray = pBuffer->mapForWriteDiscardAs< VertexPosition4fColor4f >();
	D3D11Utils_Box::writeAxisAlignedWireframeGrid( origin, size, resolution, color, pArray );
	pBuffer->unmap();

	return pBuffer;
}