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
	glGenSamplers( 1, &m_id );
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

GLint GLSamplerObject::wrapModeS() const
{
	GLint output;
	glGetSamplerParameteriv( m_id, GL_TEXTURE_WRAP_S, &output );
	return output;
}

GLint GLSamplerObject::wrapModeT() const
{
	GLint output;
	glGetSamplerParameteriv( m_id, GL_TEXTURE_WRAP_T, &output );
	return output;
}

GLint GLSamplerObject::wrapModeR() const
{
	GLint output;
	glGetSamplerParameteriv( m_id, GL_TEXTURE_WRAP_R, &output );
	return output;
}

void GLSamplerObject::setAllWrapModesClampToEdge()
{
	glSamplerParameteri( m_id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glSamplerParameteri( m_id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glSamplerParameteri( m_id, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
}
