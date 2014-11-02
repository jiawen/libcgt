#include "GLShader.h"

#include <cstdio>

#include <io/File.h>

GLShader* GLShader::vertexShaderFromSourceFile( const char* filename )
{
	return GLShader::fromFile( filename, GL_VERTEX_SHADER );
}

GLShader* GLShader::fragmentShaderFromSourceFile( const char* filename )
{
	return GLShader::fromFile( filename, GL_FRAGMENT_SHADER );
}

GLShader::~GLShader()
{
	glDeleteShader( m_id );
	m_id = 0;
}

GLuint GLShader::id() const
{
	return m_id;
}

GLenum GLShader::type() const
{
	return m_type;
}

GLShader::GLShader( int id ) :
	m_id( id )
{

}

// static
GLShader* GLShader::fromFile( const char* filename, GLenum shaderType )
{
	std::string source = File::readTextFile( filename );
	if( source.size() == 0 )
	{
		return nullptr;
	}

	// create the shader
	GLuint id = glCreateShader( shaderType );

	// copy its source over
	const char* sourceCStr = source.c_str();
	glShaderSource( id, 1, &sourceCStr, NULL );

	// compile it
	glCompileShader( id );

	// get the status and error message
	GLint status;
	glGetShaderiv( id, GL_COMPILE_STATUS, &status );

	// If it failed, then print the info log
	// and return nullptr.
	if( status == GL_FALSE )
	{
		GLint infoLogLength;
		glGetShaderiv( id, GL_INFO_LOG_LENGTH, &infoLogLength );

		std::string infoLog( infoLogLength, '\0' );
		glGetShaderInfoLog( id, static_cast< GLsizei >( infoLog.size() ),
			NULL, &( infoLog[0] ) );

		printf( "Shader compilation failed:\n%s\n", infoLog.c_str() );

		glDeleteShader( id );

		return nullptr;
	}

	return new GLShader( id );	
}
