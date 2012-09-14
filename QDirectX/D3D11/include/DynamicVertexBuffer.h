#pragma once

#include <D3D11.h>

class DynamicVertexBuffer
{
public:

	// Create a new DynamicVertexBuffer
	//
	// capacity is specified in number of vertices
	// vertexSizeBytes is the number of bytes per vertex
	// total capacity in bytes is capacity * vertexSizeBytes
	//
	// Returns nullptr if it fails
	// capacity and vertex size can be <= 0 if you want a "null" vertex buffer that can be resized later
	static DynamicVertexBuffer* create( ID3D11Device* pDevice, int capacity = 0, int vertexSizeBytes = 0 );
	virtual ~DynamicVertexBuffer();

	// A vertex buffer is null if its capacity or vertex size is <= 0
	bool isNull() const;
	bool notNull() const;

	int capacity() const;
	int vertexSizeBytes() const;

	// destroys all existing data
	HRESULT resize( int capacity );
	// also changes the vertex size
	HRESULT resize( int capacity, int vertexSizeBytes );

	ID3D11Buffer* buffer();
	UINT defaultStride();
	UINT defaultOffset();

	D3D11_MAPPED_SUBRESOURCE mapForWriteDiscard();

	// same as reinterpret_cast< T* >( mapForWriteDiscard().pData )
	template< typename T >
	T* mapForWriteDiscardAs()
	{
		return reinterpret_cast< T* >( mapForWriteDiscard().pData );
	}

	void unmap();	

private:

	DynamicVertexBuffer( ID3D11Device* pDevice );

	int m_capacity;
	int m_vertexSizeBytes;

	ID3D11Buffer* m_pBuffer;
	ID3D11Device* m_pDevice;
	ID3D11DeviceContext* m_pContext;
	
};
