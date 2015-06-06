#pragma once

#include <memory>

#include <GL/glew.h>


#include "GLPrimitiveType.h"
#include "GLVertexArrayObject.h"
#include "GLBufferObject.h"

template< typename T >
class GLInterleavedVertexBufferView
{
public:

    GLInterleavedVertexBufferView( int nVertices, GLPrimitiveType primitiveType,
        std::shared_ptr< GLBufferObject > pBuffer, GLintptr offsetBytes );

    Array1DView< T > mapForWrite();
    void unmap();

    // Binds the VAO, calls glDrawArrays(), then unbinds the VAO.
    void draw();

    int numVertices() const;

    GLsizeiptr numBytes() const;

private:

    GLintptr m_offsetBytes;
    GLsizeiptr m_nBytes;

    int m_nVertices;
    GLPrimitiveType m_primitiveType;

    std::shared_ptr< GLVertexArrayObject > m_pVAO;
    std::shared_ptr< GLBufferObject > m_pVBO;

};

template< typename T >
GLInterleavedVertexBufferView< T >::GLInterleavedVertexBufferView( int nVertices, GLPrimitiveType primitiveType,
    std::shared_ptr< GLBufferObject > pBuffer, GLintptr offsetBytes ) :
    m_nVertices( nVertices ),
    m_primitiveType( primitiveType ),
    m_offsetBytes( offsetBytes ),
    m_nBytes( nVertices * sizeof( T ) ),
    m_pVBO( pBuffer )
{
    m_pVAO = std::make_shared< GLVertexArrayObject >();

    for( int i = 0; i < T::s_numAttributes; ++i )
    {
        m_pVAO->enableAttribute( i );
        m_pVAO->mapAttributeIndexToBindingIndex( i, 0 );
        m_pVAO->setAttributeFormat( i, T::s_numComponents[ i ],
            GLVertexAttributeType::FLOAT, true,
            T::s_relativeOffsets[ i ] );        
    }

    m_pVAO->attachBuffer( 0, m_pVBO.get( ), m_offsetBytes, sizeof( T ) );
}

template< typename T >
Array1DView< T > GLInterleavedVertexBufferView< T >::mapForWrite()
{
    return m_pVBO->mapRangeAs< T >( m_offsetBytes, m_nBytes,
        GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT );
}

template< typename T >
void GLInterleavedVertexBufferView< T >::unmap()
{
    m_pVBO->unmap();
}

template< typename T >
int GLInterleavedVertexBufferView< T >::numVertices() const
{
    return m_nVertices;
}

template< typename T >
GLsizeiptr GLInterleavedVertexBufferView< T >::numBytes() const
{
    return m_nBytes;
}

template< typename T >
void GLInterleavedVertexBufferView< T >::draw()
{
    m_pVAO->bind();
    glDrawArrays( static_cast< GLenum >( m_primitiveType ), 0, m_nVertices );
    m_pVAO->unbindAll();
}
