#include "GLProgram.h"

#include <vecmath/libcgt_vecmath.h>

#include "GLShader.h"

#include "GLUtilities.h"

// static
GLProgram* GLProgram::fromShaders( std::vector< GLShader* > shaders )
{
	// Create the program.
	GLuint id = glCreateProgram();
	
	// Attach all the shaders.
	for( size_t i = 0; i < shaders.size(); ++i )
	{
		glAttachShader( id, shaders[i]->id() );
	}
	
    // Link.
	glLinkProgram( id );

	// If it failed, then print the info log and return nullptr.
	GLint status;
    glGetProgramiv( id, GL_LINK_STATUS, &status );
    if( status == GL_FALSE )
	{
        // infoLogLength includes the terminating '\0'.
		GLint infoLogLength;
		glGetProgramiv( id, GL_INFO_LOG_LENGTH, &infoLogLength );

		std::string infoLog( infoLogLength, '\0' );
		glGetProgramInfoLog( id, static_cast< GLsizei >( infoLog.size() ),
			NULL, &( infoLog[0] ) );

		printf( "Program linking failed:\n%s\n", infoLog.c_str() );

		glDeleteProgram( id );

		return nullptr;
	}

	return new GLProgram( id );
}

// static
void GLProgram::disableAll()
{
	glUseProgram( 0 );
}

GLProgram::~GLProgram()
{
	glDeleteProgram( m_id );
}

GLuint GLProgram::id() const
{
	return m_id;
}

GLint GLProgram::numActiveUniforms() const
{
	GLint numActiveUniforms;
	glGetProgramiv( m_id, GL_ACTIVE_UNIFORMS, &numActiveUniforms );
	return numActiveUniforms;
}

GLint GLProgram::uniformLocation( const GLchar* name ) const
{
	return glGetUniformLocation( m_id, name );
}

Matrix4f GLProgram::getUniformMatrix4f( GLint uniformLocation )
{
    Matrix4f m;
    glGetUniformfv( m_id, uniformLocation, m );
    return m;
}

void GLProgram::setUniformMatrix4f( GLint uniformLocation, const Matrix4f& matrix )
{
	glProgramUniformMatrix4fv( m_id, uniformLocation, 1, false, matrix );
}

void GLProgram::setUniformInt( GLint uniformLocation, int x )
{
	glProgramUniform1i( m_id, uniformLocation, x );
}

void GLProgram::setUniformFloat( GLint uniformLocation, float x )
{
	glProgramUniform1f( m_id, uniformLocation, x );
}

void GLProgram::setUniformFloat( const GLchar* name, float x )
{
    setUniformFloat( uniformLocation( name ), x );
}

void GLProgram::setUniformMatrix4f( const GLchar* name, const Matrix4f& matrix )
{
    setUniformMatrix4f( uniformLocation( name ), matrix );
}

void GLProgram::setUniformVector2f( const GLchar* name, float x, float y )
{    
    glProgramUniform2f( m_id, uniformLocation( name ), x, y );
}

void GLProgram::setUniformVector2f( const GLchar* name, const Vector2f& v )
{
    glProgramUniform2fv( m_id, uniformLocation( name ), 1, v );
}

void GLProgram::setUniformVector3f( const GLchar* name, float x, float y, float z )
{
    glProgramUniform3f( m_id, uniformLocation( name ), x, y, z );
}

void GLProgram::setUniformVector3f( const GLchar* name, const Vector3f& v )
{
    glProgramUniform3fv( m_id, uniformLocation( name ), 1, v );
}

void GLProgram::setUniformVector4f( const GLchar* name, float x, float y, float z, float w )
{
    glProgramUniform4f( m_id, uniformLocation( name ), x, y, z, w );
}

void GLProgram::setUniformVector4f( const GLchar* name, const Vector4f& v )
{
    glProgramUniform4fv( m_id, uniformLocation( name ), 1, v );
}

void GLProgram::setUniformInt( const GLchar* name, int x )
{
    setUniformInt( uniformLocation( name ), x );
}

void GLProgram::setUniformVector2i( const GLchar* name, int x, int y )
{
    glProgramUniform2i( m_id, uniformLocation( name ), x, y );
}

void GLProgram::setUniformVector2i( const GLchar* name, const Vector2i& v )
{
    glProgramUniform2iv( m_id, uniformLocation( name ), 1, v );
}

void GLProgram::setUniformVector3i( const GLchar* name, int x, int y, int z )
{
    glProgramUniform3i( m_id, uniformLocation( name ), x, y, z );
}

void GLProgram::setUniformVector3i( const GLchar* name, const Vector3i& v )
{
    glProgramUniform3iv( m_id, uniformLocation( name ), 1, v );
}

void GLProgram::setUniformVector4i( const GLchar* name, int x, int y, int z, int w )
{
    glProgramUniform4i( m_id, uniformLocation( name ), x, y, z, w );
}

void GLProgram::setUniformVector4i( const GLchar* name, const Vector4i& v )
{
    glProgramUniform4iv( m_id, uniformLocation( name ), 1, v );
}

void GLProgram::use()
{
    glUseProgram( m_id );
}

GLProgram::GLProgram( GLuint id ) :
	m_id( id )
{
}
