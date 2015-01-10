#include "GLVertexArrayObject.h"

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
    // TODO: ARB_DSA: use glEnableVertexAttribArray() / glDisableVertexAttribArray().
	glEnableVertexArrayAttribEXT( m_id, attributeIndex );
}

void GLVertexArrayObject::disableAttribute( GLuint attributeIndex )
{
	glDisableVertexArrayAttribEXT( m_id, attributeIndex );
}

void GLVertexArrayObject::mapAttributeIndexToBindingIndex( GLuint attributeIndex,
    GLuint bindingIndex )
{
    // TODO: ARB_DSA: glVertexArrayVertexAttribBinding()
    glVertexArrayVertexAttribBindingEXT( m_id, attributeIndex, bindingIndex );
}

void GLVertexArrayObject::setAttributeFormat( GLuint attributeIndex, GLint nComponents,
    GLVertexAttributeType type, bool normalized, GLuint relativeOffsetBytes )
{
    glVertexArrayVertexAttribFormatEXT( m_id, attributeIndex, nComponents,
        static_cast< GLint >( type ), normalized, relativeOffsetBytes );
}

void GLVertexArrayObject::setAttributeIntegerFormat( GLuint attributeIndex, GLint nComponents,
    GLVertexAttributeType type, GLuint relativeOffsetBytes )
{
    glVertexArrayVertexAttribIFormatEXT( m_id, attributeIndex, nComponents,
        static_cast< GLint >( type ), relativeOffsetBytes );
}

void GLVertexArrayObject::setAttributeDoubleFormat( GLuint attributeIndex, GLint nComponents,
    GLuint relativeOffsetBytes )
{
    glVertexArrayVertexAttribIFormatEXT( m_id, attributeIndex, nComponents,
        GL_DOUBLE, relativeOffsetBytes );
}

void GLVertexArrayObject::attachBuffer( GLuint bindingIndex, GLBufferObject* pBuffer,
        GLintptr offset, GLsizei stride )
{
    glVertexArrayBindVertexBufferEXT( m_id, bindingIndex,
        pBuffer->id(), offset, stride );

    // TODO: --> OpenGL wiki (https://www.opengl.org/wiki/Vertex_Specification)
    // stride of 0 is broken, no error reported

    // TODO: ARB_DSA:
    // glVertexArrayVertexBuffer(), glVertexArrayVertexBuffers()
}

void GLVertexArrayObject::detachBuffer( GLuint bindingIndex )
{
    glVertexArrayBindVertexBufferEXT( m_id, bindingIndex, 0, 0, 0 );
}