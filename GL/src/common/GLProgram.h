#pragma once

#include <vector>

#ifdef GL_PLATFORM_ES_31
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#endif
#ifdef GL_PLATFORM_45
#include <GL/glew.h>
#endif

class GLShader;
class Matrix4f;
class Vector2f;
class Vector3f;
class Vector4f;
class Vector2i;
class Vector3i;
class Vector4i;

// TODO: replace this with GLSeparableProgram.

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

    void setUniformInt( GLint uniformLocation, int x );
    void setUniformFloat( GLint uniformLocation, float x );
    void setUniformMatrix4f( GLint uniformLocation, const Matrix4f& m );
    void setUniformVector2f( GLint uniformLocation, const Vector2f& v );
    void setUniformVector2i( GLint uniformLocation, const Vector2i& v );
    void setUniformVector3f( GLint uniformLocation, const Vector3f& v );
    void setUniformVector3i( GLint uniformLocation, const Vector3i& v );
    void setUniformVector4f( GLint uniformLocation, const Vector4f& v );
    void setUniformVector4i( GLint uniformLocation, const Vector4i& v );

    void setUniformInt( const GLchar* name, int x );
    void setUniformFloat( const GLchar* name, float x );
    void setUniformMatrix4f( const GLchar* name, const Matrix4f& m );
    void setUniformVector2f( const GLchar* name, const Vector2f& v );
    void setUniformVector3f( const GLchar* name, const Vector3f& v );
    void setUniformVector4f( const GLchar* name, const Vector4f& v );
    void setUniformVector2i( const GLchar* name, const Vector2i& v );
    void setUniformVector3i( const GLchar* name, const Vector3i& v );
    void setUniformVector4i( const GLchar* name, const Vector4i& v );

    void use();

private:

    GLProgram( GLuint id );

    GLuint m_id;
};
