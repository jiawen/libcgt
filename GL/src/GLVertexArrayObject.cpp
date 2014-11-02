#include "GLVertexArrayObject.h"

// static
void GLVertexArrayObject::unbind()
{
	glBindVertexArray( 0 );
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

GLuint GLVertexArrayObject::id()
{
	return m_id;
}
	
void GLVertexArrayObject::bind()
{
	glBindVertexArray( m_id );
}

void GLVertexArrayObject::enableAttribute( GLuint index )
{
	glEnableVertexArrayAttribEXT( m_id, index );
}

void GLVertexArrayObject::disableAttribute( GLuint index )
{
	glDisableVertexArrayAttribEXT( m_id, index );
}