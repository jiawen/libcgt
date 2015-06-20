#include "DynamicVertexBuffer.h"

// static
DynamicVertexBuffer* DynamicVertexBuffer::create( ID3D11Device* pDevice, int capacity, int vertexSizeBytes )
{
    DynamicVertexBuffer* pBuffer = new DynamicVertexBuffer( pDevice );
    HRESULT hr = pBuffer->resize( capacity, vertexSizeBytes );
    if( SUCCEEDED( hr ) )
    {
        return pBuffer;
    }
    else
    {
        delete pBuffer;
        return nullptr;
    }
}

// virtual
DynamicVertexBuffer::~DynamicVertexBuffer()
{
    if( m_pBuffer != nullptr )
    {
        m_pBuffer->Release();
        m_pBuffer = nullptr;
    }

    m_pContext->Release();
    m_pContext = nullptr;
    m_pDevice->Release();
    m_pDevice = nullptr;
}

bool DynamicVertexBuffer::isNull() const
{
    return( capacity() <= 0 || vertexSizeBytes() <= 0 );
}

bool DynamicVertexBuffer::notNull() const
{
    return !isNull();
}

int DynamicVertexBuffer::capacity() const
{
    return m_capacity;
}

int DynamicVertexBuffer::vertexSizeBytes() const
{
    return m_vertexSizeBytes;
}

HRESULT DynamicVertexBuffer::resize( int capacity )
{
    return resize( capacity, vertexSizeBytes() );
}

HRESULT DynamicVertexBuffer::resize( int capacity, int vertexSizeBytes )
{
    // if the sizes are equal, then do nothing
    if( capacity == m_capacity &&
        vertexSizeBytes == m_vertexSizeBytes )
    {
        return S_FALSE;
    }

    // delete the old one
    if( m_pBuffer != nullptr )
    {
        m_pBuffer->Release();
        m_pBuffer = nullptr;
    }

    m_capacity = capacity;
    m_vertexSizeBytes = vertexSizeBytes;

    if( notNull() )
    {
        int bufferSize = m_capacity * m_vertexSizeBytes;

        D3D11_BUFFER_DESC bd;
        bd.ByteWidth = bufferSize;
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        bd.MiscFlags = 0;

        HRESULT hr = m_pDevice->CreateBuffer( &bd, NULL, &m_pBuffer );
        return hr;
    }
    else
    {
        return S_FALSE;
    }
}

ID3D11Buffer* DynamicVertexBuffer::buffer()
{
    return m_pBuffer;
}

UINT DynamicVertexBuffer::defaultStride()
{
    return m_vertexSizeBytes;
}

UINT DynamicVertexBuffer::defaultOffset()
{
    return 0;
}

D3D11_MAPPED_SUBRESOURCE DynamicVertexBuffer::mapForWriteDiscard()
{
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    m_pContext->Map( m_pBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource );
    return mappedResource;
}

void DynamicVertexBuffer::unmap()
{
    m_pContext->Unmap( m_pBuffer, 0 );
}

DynamicVertexBuffer::DynamicVertexBuffer( ID3D11Device* pDevice ) :

    m_capacity( 0 ),
    m_vertexSizeBytes( 0 ),

    m_pBuffer( nullptr ),
    m_pDevice( pDevice )

{
    pDevice->GetImmediateContext( &m_pContext );
}
