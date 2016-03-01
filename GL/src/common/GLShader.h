#pragma once

#ifdef GL_PLATFORM_ES_31
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#endif
#ifdef GL_PLATFORM_45
#include <GL/glew.h>
#endif

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
