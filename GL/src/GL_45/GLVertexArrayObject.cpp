#include "GLVertexArrayObject.h"

#include <cassert>

#include <common/Array1D.h>
#include "GLBufferObject.h"

// static
GLuint GLVertexArrayObject::boundId()
{
    int bid;
    glGetIntegerv( GL_VERTEX_ARRAY_BINDING, &bid );
    return bid;
}

// static
void GLVertexArrayObject::unbindAll()
{
    glBindVertexArray( 0 );
}

// static
int GLVertexArrayObject::maxNumVertexAttributes()
{
    int n;
    glGetIntegerv( GL_MAX_VERTEX_ATTRIBS, &n );
    return n;
}

// static
int GLVertexArrayObject::maxNumVertexAttributeBindings()
{
    int n;
    glGetIntegerv( GL_MAX_VERTEX_ATTRIB_BINDINGS, &n );
    return n;
}

// static
int GLVertexArrayObject::maxVertexAttributeStride()
{
    int n;
    glGetIntegerv( GL_MAX_VERTEX_ATTRIB_STRIDE, &n );
    return n;
}

GLVertexArrayObject::GLVertexArrayObject() :
    m_id( 0 )
{
    glCreateVertexArrays( 1, &m_id );
}

// virtual
GLVertexArrayObject::~GLVertexArrayObject()
{
    glDeleteVertexArrays( 1, &m_id );
}

GLuint GLVertexArrayObject::id() const
{
    return m_id;
}

void GLVertexArrayObject::bind()
{
    glBindVertexArray( m_id );
}

void GLVertexArrayObject::enableAttribute( GLuint attributeIndex )
{
    glEnableVertexArrayAttrib( m_id, attributeIndex );
}

void GLVertexArrayObject::disableAttribute( GLuint attributeIndex )
{
    glDisableVertexArrayAttrib( m_id, attributeIndex );
}

void GLVertexArrayObject::mapAttributeIndexToBindingIndex(
    GLuint attributeIndex, GLuint bindingIndex )
{
    glVertexArrayAttribBinding( m_id, attributeIndex, bindingIndex );
}

void GLVertexArrayObject::setAttributeFormat( GLuint attributeIndex,
    GLint nComponents, GLVertexAttributeType type, bool normalized,
    GLuint relativeOffsetBytes )
{
    glVertexArrayAttribFormat( m_id, attributeIndex, nComponents,
        glVertexAttributeType( type ), normalized,
        relativeOffsetBytes
    );
}

void GLVertexArrayObject::setAttributeIntegerFormat( GLuint attributeIndex,
    GLint nComponents, GLVertexAttributeType type, GLuint relativeOffsetBytes )
{
    glVertexArrayAttribIFormat( m_id, attributeIndex, nComponents,
        glVertexAttributeType( type ), relativeOffsetBytes
    );
}

void GLVertexArrayObject::setAttributeDoubleFormat( GLuint attributeIndex,
    GLint nComponents, GLuint relativeOffsetBytes )
{
    glVertexArrayAttribLFormat( m_id, attributeIndex, nComponents,
        GL_DOUBLE, relativeOffsetBytes );
}

void GLVertexArrayObject::attachBuffer( GLuint bindingIndex,
    GLBufferObject* pBuffer, GLintptr offset, GLsizei stride )
{
    assert( stride != 0 );
    assert( stride <= maxVertexAttributeStride() );
    glVertexArrayVertexBuffer( m_id, bindingIndex,
        pBuffer->id(), offset, stride );
}

void GLVertexArrayObject::attachBuffers( GLuint firstBindingIndex,
    Array1DReadView< GLBufferObject* > buffers,
    Array1DReadView< GLintptr > offsets,
    Array1DReadView< GLsizei > strides )
{
    size_t count = buffers.size();
    Array1D< GLuint > ids( count );
    for( size_t i = 0; i < count; ++i )
    {
        ids[ i ] = buffers[ i ]->id();
    }
    glVertexArrayVertexBuffers( m_id, firstBindingIndex,
        static_cast< GLsizei >( count ), ids, offsets.pointer(),
        strides.pointer()
    );
}

void GLVertexArrayObject::attachIndexBuffer( GLBufferObject* pBuffer )
{
    glVertexArrayElementBuffer( m_id, pBuffer->id() );
}

int GLVertexArrayObject::getAttachedIndexBufferId()
{
    GLint bufferId;
    glGetVertexArrayiv( m_id, GL_ELEMENT_ARRAY_BUFFER_BINDING, &bufferId );
    return bufferId;
}

void GLVertexArrayObject::detachBuffer( GLuint bindingIndex )
{
    glVertexArrayVertexBuffer( m_id, bindingIndex, 0, 0, 0 );
}

void GLVertexArrayObject::detachBuffers( GLuint firstBindingIndex, int count )
{
    Array1D< GLuint > ids( count, sizeof( GLuint), 0 );
    Array1D< GLintptr > offsets( count, sizeof( GLintptr ), 0 );
    Array1D< GLsizei > strides( count, sizeof( int ), 0 );
    glVertexArrayVertexBuffers( m_id, firstBindingIndex, count, ids,
        offsets.pointer(), strides );
}
