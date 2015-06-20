#include "GLSamplerObject.h"

// static
GLfloat GLSamplerObject::getLargestSupportedAnisotropy()
{
    GLfloat largestSupportedAnisotropy;
    glGetFloatv( GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &largestSupportedAnisotropy );
    return largestSupportedAnisotropy;
}

// static
void GLSamplerObject::unbind( GLuint textureUnit )
{
    glBindSampler( textureUnit, 0 );
}

GLSamplerObject::GLSamplerObject()
{
    glCreateSamplers( 1, &m_id );
}

// virtual
GLSamplerObject::~GLSamplerObject()
{
    glDeleteSamplers( 1, &m_id );
}

GLuint GLSamplerObject::id() const
{
    return m_id;
}

void GLSamplerObject::bind( GLuint textureUnit )
{
    glBindSampler( textureUnit, m_id );
}

GLTextureFilterMode GLSamplerObject::minFilterMode() const
{
    GLint output;
    glGetSamplerParameteriv( m_id, GL_TEXTURE_MIN_FILTER, &output );
    return static_cast< GLTextureFilterMode >( output );
}

GLTextureFilterMode GLSamplerObject::magFilterMode() const
{
    GLint output;
    glGetSamplerParameteriv( m_id, GL_TEXTURE_MAG_FILTER, &output );
    return static_cast< GLTextureFilterMode >( output );
}

GLfloat GLSamplerObject::anisotropy() const
{
    GLfloat output;
    glGetSamplerParameterfv( m_id, GL_TEXTURE_MAX_ANISOTROPY_EXT, &output );
    return output;
}

void GLSamplerObject::setMinFilterMode( GLTextureFilterMode mode )
{
    glSamplerParameteri( m_id, GL_TEXTURE_MIN_FILTER, static_cast< GLint >( mode )  );
}

void GLSamplerObject::setMagFilterMode( GLTextureFilterMode mode )
{
    glSamplerParameteri( m_id, GL_TEXTURE_MAG_FILTER, static_cast< GLint >( mode ) );
}

void GLSamplerObject::setMinMagFilterMode( GLTextureFilterMode minMode, GLTextureFilterMode magMode )
{
    glSamplerParameteri( m_id, GL_TEXTURE_MIN_FILTER, static_cast< GLint >( minMode ) );
    glSamplerParameteri( m_id, GL_TEXTURE_MAG_FILTER, static_cast< GLint >( magMode ) );
}

void GLSamplerObject::setAnisotropy( GLfloat anisotropy )
{
    glSamplerParameterf( m_id, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy );
}

void GLSamplerObject::setSWrapMode( GLWrapMode mode )
{
    glSamplerParameteri( m_id, GL_TEXTURE_WRAP_S, static_cast< GLint >( mode ) );
}

void GLSamplerObject::setTWrapMode( GLWrapMode mode )
{
    glSamplerParameteri( m_id, GL_TEXTURE_WRAP_T, static_cast< GLint >( mode ) );
}

void GLSamplerObject::setRWrapMode( GLWrapMode mode )
{
    glSamplerParameteri( m_id, GL_TEXTURE_WRAP_R, static_cast< GLint >( mode ) );
}

void GLSamplerObject::setSTWrapModes( GLWrapMode sMode, GLWrapMode tMode )
{
    glSamplerParameteri( m_id, GL_TEXTURE_WRAP_S, static_cast< GLint >( sMode ) );
    glSamplerParameteri( m_id, GL_TEXTURE_WRAP_T, static_cast< GLint >( sMode ) );
}

void GLSamplerObject::setSTWrapModes( GLWrapMode mode )
{
    glSamplerParameteri( m_id, GL_TEXTURE_WRAP_S, static_cast< GLint >( mode ) );
    glSamplerParameteri( m_id, GL_TEXTURE_WRAP_T, static_cast< GLint >( mode ) );
}

void GLSamplerObject::setWrapModes( GLWrapMode sMode, GLWrapMode tMode, GLWrapMode rMode )
{
    glSamplerParameteri( m_id, GL_TEXTURE_WRAP_S, static_cast< GLint >( sMode ) );
    glSamplerParameteri( m_id, GL_TEXTURE_WRAP_T, static_cast< GLint >( sMode ) );
    glSamplerParameteri( m_id, GL_TEXTURE_WRAP_R, static_cast< GLint >( sMode ) );
}

void GLSamplerObject::setWrapModes( GLWrapMode mode )
{
    glSamplerParameteri( m_id, GL_TEXTURE_WRAP_S, static_cast< GLint >( mode ) );
    glSamplerParameteri( m_id, GL_TEXTURE_WRAP_T, static_cast< GLint >( mode ) );
    glSamplerParameteri( m_id, GL_TEXTURE_WRAP_R, static_cast< GLint >( mode ) );
}

GLWrapMode GLSamplerObject::wrapModeS() const
{
    GLint output;
    glGetSamplerParameteriv( m_id, GL_TEXTURE_WRAP_S, &output );
    return static_cast< GLWrapMode >( output );
}

GLWrapMode GLSamplerObject::wrapModeT() const
{
    GLint output;
    glGetSamplerParameteriv( m_id, GL_TEXTURE_WRAP_T, &output );
    return static_cast< GLWrapMode >( output );
}

GLWrapMode GLSamplerObject::wrapModeR() const
{
    GLint output;
    glGetSamplerParameteriv( m_id, GL_TEXTURE_WRAP_R, &output );
    return static_cast< GLWrapMode >( output );
}
