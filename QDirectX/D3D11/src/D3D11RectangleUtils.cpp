#include "D3D11RectangleUtils.h"

#include <d3d11.h>

#include "DynamicVertexBuffer.h"
#include "VertexPosition4f.h"
#include "VertexPosition4fTexture2f.h"

// static
DynamicVertexBuffer* D3D11RectangleUtils::createScreenAlignedRectangle( ID3D11Device* pDevice,
	float x, float y, float width, float height,
	bool uv00AtTopLeft, float z, float w )
{
	DynamicVertexBuffer* pBuffer = DynamicVertexBuffer::create( pDevice, 6, VertexPosition4fTexture2f::sizeInBytes() );

	VertexPosition4fTexture2f* vertexArray = pBuffer->mapForWriteDiscardAs< VertexPosition4fTexture2f >();
	writeScreenAlignedRectangle( vertexArray, x, y, width, height, uv00AtTopLeft, z, w );
	pBuffer->unmap();

	return pBuffer;
}

// static
void D3D11RectangleUtils::writeScreenAlignedRectangle( VertexPosition4f* vertexArray,
	float x, float y, float width, float height,
	float z, float w )
{
	writeScreenAlignedRectanglePositions( vertexArray, x, y, width, height, z, w );
}

// static
void D3D11RectangleUtils::writeScreenAlignedRectangle( VertexPosition4fTexture2f* vertexArray,
	float x, float y, float width, float height,
	bool uv00AtTopLeft, float z, float w )
{
	writeScreenAlignedRectanglePositions( vertexArray, x, y, width, height, z, w );
	writeScreenAlignedRectangleTextureCoordinates( vertexArray, uv00AtTopLeft );
}