#ifndef DYNAMIC_VERTEX_BUFFER_H
#define DYNAMIC_VERTEX_BUFFER_H

#include <D3D11.h>

class DynamicVertexBuffer
{
public:

	DynamicVertexBuffer( ID3D11Device* pDevice, int capacity, int vertexSizeBytes );
	virtual ~DynamicVertexBuffer();

	int capacity();

	ID3D11Buffer* buffer();
	UINT defaultStride();
	UINT defaultOffset();

	D3D11_MAPPED_SUBRESOURCE mapForWriteDiscard();
	void unmap();

private:

	int m_capacity;
	int m_vertexSizeBytes;
	
	ID3D11Buffer* m_pBuffer;
	ID3D11DeviceContext* m_pContext;
	
};

#endif // DYNAMIC_VERTEX_BUFFER_H
