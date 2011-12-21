#ifndef STAGING_STRUCTURED_BUFFER_H
#define STAGING_STRUCTURED_BUFFER_H

#include <D3D11.h>

class StagingStructuredBuffer
{
public:

	static StagingStructuredBuffer* create( ID3D11Device* pDevice,
		int nElements, int elementSizeBytes );

	virtual ~StagingStructuredBuffer();

	int numElements() const;
	int elementSizeBytes() const;

	ID3D11Buffer* buffer() const;
	
	// read/write mapping
	D3D11_MAPPED_SUBRESOURCE mapForReadWrite();
	void unmap();

	// Copy from pSource to this
	void copyFrom( ID3D11Buffer* pSource );

	// Copy from this to pTarget
	void copyTo( ID3D11Buffer* pTarget );

private:

	StagingStructuredBuffer( ID3D11Device* pDevice,
		int nElements, int elementSizeBytes,
		ID3D11Buffer* pBuffer );

	int m_nElements;
	int m_elementSizeBytes;

	ID3D11Device* m_pDevice;
	ID3D11Buffer* m_pBuffer;
	ID3D11DeviceContext* m_pContext;
};

#endif // STAGING_STRUCTURED_BUFFER_H
