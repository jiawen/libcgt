#pragma once

#include <vector>

#include <GL/glew.h>

class GLShader;
class Matrix4f;
class Vector2f;
class Vector3f;
class Vector4f;
class Vector2i;
class Vector3i;
class Vector4i;

class GLProgram
{
public:

	static GLProgram* fromShaders( std::vector< GLShader* > shaders );
	static void disableAll();
	
	virtual ~GLProgram();

	GLuint id() const;

	GLint numActiveUniforms() const;
	GLint uniformLocation( const GLchar* name ) const;
  	
	Matrix4f getUniformMatrix4f( GLint uniformLocation );
		
	void setUniformMatrix4f( GLint uniformLocation, const Matrix4f& matrix );

	void setUniformInt( GLint uniformLocation, int x );
	void setUniformFloat( GLint uniformLocation, float x );

	void setUniformFloat( const GLchar* name, float x );
	void setUniformVector2f( const GLchar* name, float x, float y );
	void setUniformVector2f( const GLchar* name, const Vector2f& v );
	void setUniformVector3f( const GLchar* name, float x, float y, float z );
	void setUniformVector3f( const GLchar* name, const Vector3f& v );
	void setUniformVector4f( const GLchar* name, float x, float y, float z, float w );
	void setUniformVector4f( const GLchar* name, const Vector4f& v );

	void setUniformInt( const GLchar* name, int x );
    void setUniformMatrix4f( const GLchar* name, const Matrix4f& matrix );
	void setUniformVector2i( const GLchar* name, int x, int y );
	void setUniformVector2i( const GLchar* name, const Vector2i& v );
	void setUniformVector3i( const GLchar* name, int x, int y, int z );
	void setUniformVector3i( const GLchar* name, const Vector3i& v );
	void setUniformVector4i( const GLchar* name, int x, int y, int z, int w );
	void setUniformVector4i( const GLchar* name, const Vector4i& v );

	void use();

private:

	GLProgram( GLuint id );
	
	GLuint m_id;	
};
