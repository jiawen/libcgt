#pragma once

#include <string>

#ifdef GL_PLATFORM_ES_31
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#endif
#ifdef GL_PLATFORM_45
#include <GL/glew.h>
#endif

class Matrix4f;
class Vector2f;
class Vector3f;
class Vector4f;
class Vector2i;
class Vector3i;
class Vector4i;

class GLSeparableProgram
{
public:

    enum class Type
    {
        NO_TYPE = 0,
        COMPUTE_SHADER = GL_COMPUTE_SHADER,
        FRAGMENT_SHADER = GL_FRAGMENT_SHADER,
#ifdef GL_PLATFORM_45
        GEOMETRY_SHADER = GL_GEOMETRY_SHADER,
        TESS_CONTROL_SHADER = GL_TESS_CONTROL_SHADER,
        TESS_EVALUATION_SHADER = GL_TESS_EVALUATION_SHADER,
#endif
        VERTEX_SHADER = GL_VERTEX_SHADER
    };

    static GLSeparableProgram fromFile( Type shaderType,
        const char* filename );

    // The default invalid program (with id = 0).
    GLSeparableProgram() = default;

    // Construct a separable GL shader program given source code (as a string).
    // If it successfully compiled, isValid() will be true.
    // You can check the compilation info log with infoLog().
    GLSeparableProgram( Type shaderType, const char* sourceCode );

    // TODO(VS2015): = default
    GLSeparableProgram( GLSeparableProgram&& move );
    GLSeparableProgram& operator = ( GLSeparableProgram&& move );

    virtual ~GLSeparableProgram();

    GLuint id() const;

    std::string infoLog() const;

    // Returns true if this program is valid (id() != 0).
    bool isValid() const;

    Type type() const;

    void setUniformInt( GLint uniformLocation, int x );
    void setUniformFloat( GLint uniformLocation, float x );
    void setUniformMatrix4f( GLint uniformLocation, const Matrix4f& m );
    void setUniformVector2f( GLint uniformLocation, const Vector2f& v );
    void setUniformVector2i( GLint uniformLocation, const Vector2i& v );
    void setUniformVector3f( GLint uniformLocation, const Vector3f& v );
    void setUniformVector3i( GLint uniformLocation, const Vector3i& v );
    void setUniformVector4f( GLint uniformLocation, const Vector4f& v );
    void setUniformVector4i( GLint uniformLocation, const Vector4i& v );

    // Retrieves the location of a uniform of the given name.
    GLint uniformLocation( const GLchar* name ) const;

private:

    GLuint m_id = 0;
    Type m_type = Type::NO_TYPE;

    void destroy();
};
