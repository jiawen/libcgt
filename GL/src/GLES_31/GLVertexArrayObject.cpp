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
    glGenVertexArrays( 1, &m_id );
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
    glEnableVertexAttribArray( attributeIndex );
}

void GLVertexArrayObject::disableAttribute( GLuint attributeIndex )
{
    glDisableVertexAttribArray( attributeIndex );
}

void GLVertexArrayObject::mapAttributeIndexToBindingIndex(
    GLuint attributeIndex, GLuint bindingIndex )
{
    glVertexAttribBinding( attributeIndex, bindingIndex );
}

void GLVertexArrayObject::setAttributeFormat( GLuint attributeIndex,
    GLint nComponents, GLVertexAttributeType type, bool normalized,
    GLuint relativeOffsetBytes )
{
    glVertexAttribFormat( attributeIndex, nComponents,
        glVertexAttributeType( type ), normalized, relativeOffsetBytes );
}

void GLVertexArrayObject::setAttributeIntegerFormat( GLuint attributeIndex,
    GLint nComponents, GLVertexAttributeType type, GLuint relativeOffsetBytes )
{
    glVertexAttribIFormat( attributeIndex, nComponents,
        glVertexAttributeType( type ), relativeOffsetBytes );
}

void GLVertexArrayObject::attachBuffer( GLuint bindingIndex,
    GLBufferObject* pBuffer, GLintptr offset, GLsizei stride )
{
    assert( stride != 0 );
    assert( stride <= maxVertexAttributeStride() );
    glBindVertexBuffer( bindingIndex, pBuffer->id(), offset, stride );
}

void GLVertexArrayObject::detachBuffer( GLuint bindingIndex )
{
    glBindVertexBuffer( bindingIndex, 0, 0, 0 );
}