#include "GLProgram.h"

#include <vecmath/libcgt_vecmath.h>

#include "GLShader.h"
#include "GLTexture.h"

GLProgram::GLProgram() :
	m_bIsLinked( false )
{
	m_iProgramHandle = glCreateProgram();
}

GLProgram::~GLProgram()
{
	glDeleteProgram( m_iProgramHandle );
}

GLhandleARB GLProgram::getHandle()
{
	return m_iProgramHandle;
}

void GLProgram::attachShader( GLShader* pShader )
{
	glAttachShader( m_iProgramHandle, pShader->getHandle() );
}

void GLProgram::detachShader( GLShader* pShader )
{
	glDetachShader( m_iProgramHandle, pShader->getHandle() );
}

GLint GLProgram::getNumActiveUniforms()
{
	GLint numActiveUniforms;
	glGetProgramiv( m_iProgramHandle, GL_ACTIVE_UNIFORMS, &numActiveUniforms );
	return numActiveUniforms;
}

GLint GLProgram::getUniformLocation( const GLchar* szName )
{
	return glGetUniformLocation( m_iProgramHandle, szName );
}

Matrix4f GLProgram::getUniformMatrix4f( GLint iUniformLocation )
{
    Matrix4f m;
    glGetUniformfv( m_iProgramHandle, iUniformLocation, m );
    return m;
}

void GLProgram::setUniformMatrix4f( GLint iUniformLocation, Matrix4f* pMatrix )
{
	glUniformMatrix4fv( iUniformLocation, 1, false, *pMatrix );
}

void GLProgram::setUniformSampler( GLint iUniformLocation, GLTexture* pTexture )
{
	glUniform1i( iUniformLocation, 0 /*pTexture->getTextureUnit()*/ );
}

void GLProgram::setUniformSampler( const GLchar* variableName, GLTexture* pTexture )
{
	// TODO: do error checking
    setUniformSampler( getUniformLocation( variableName ), pTexture );
}

void GLProgram::setUniformVector2f( const GLchar* variableName, float x, float y )
{    
    glUniform2f( getUniformLocation( variableName ), x, y );
}

void GLProgram::setUniformVector2f( const GLchar* variableName, const Vector2f& v )
{
    glUniform2fv( getUniformLocation( variableName ), 1, v );
}

void GLProgram::setUniformVector3f( const GLchar* variableName, float x, float y, float z )
{
    glUniform3f( getUniformLocation( variableName ), x, y, z );
}

void GLProgram::setUniformVector3f( const GLchar* variableName, const Vector3f& v )
{
    glUniform3fv( getUniformLocation( variableName ), 1, v );
}

bool GLProgram::link()
{
	GLint status;

	glLinkProgram( m_iProgramHandle );
	glGetProgramiv( m_iProgramHandle, GL_OBJECT_LINK_STATUS_ARB, &status );

	if( status == GL_TRUE )
	{
		m_bIsLinked = true;
	}
	return m_bIsLinked;
}

bool GLProgram::isLinked()
{
	return m_bIsLinked;
}

bool GLProgram::use()
{
	if( m_bIsLinked )
	{
		glUseProgram( m_iProgramHandle );
	}
	return m_bIsLinked;
}

// static
void GLProgram::disableAll()
{
	glUseProgram( 0 );
}
