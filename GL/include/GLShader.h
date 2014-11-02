#pragma once

#include <GL/glew.h>

class GLShader
{
public:

	// Factory constructors
	static GLShader* vertexShaderFromSourceFile( const char* filename );
	static GLShader* fragmentShaderFromSourceFile( const char* filename );

	// Destructor
	virtual ~GLShader();

	GLuint id() const;
	GLenum type() const;

private:

	GLuint m_id;
	GLenum m_type;

	GLShader( int id );
	static GLShader* fromFile( const char* filename, GLenum shaderType );	
};
