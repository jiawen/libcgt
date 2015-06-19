#include "GLRenderbufferObject.h"

GLRenderbufferObject::GLRenderbufferObject( const Vector2i& size,
    GLImageInternalFormat internalFormat ) :
    m_size( size ),
    m_internalFormat( internalFormat )
{
    glCreateRenderbuffers( 1, &m_id );
    glNamedRenderbufferStorage( m_id, static_cast< GLenum >( internalFormat ),
        size.x, size.y );
}

// virtual
GLRenderbufferObject::~GLRenderbufferObject()
{
    glDeleteRenderbuffers( 1, &m_id );
    m_id = 0;
}

GLuint GLRenderbufferObject::id() const
{
    return m_id;
}