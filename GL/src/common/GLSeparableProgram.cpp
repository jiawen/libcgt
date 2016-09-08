#include "GLSeparableProgram.h"

#include <cassert>

#include <io/File.h>
#include <vecmath/Matrix4f.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector3i.h>
#include <vecmath/Vector4f.h>
#include <vecmath/Vector4i.h>

// static
GLSeparableProgram GLSeparableProgram::fromFile(
    GLSeparableProgram::Type shaderType, const char* filename )
{
    return GLSeparableProgram( shaderType,
        File::readTextFile( filename ).c_str() );
}

GLSeparableProgram::GLSeparableProgram( GLSeparableProgram::Type shaderType,
    const char* sourceCode, std::string* infoLog )
{
    GLenum type = static_cast< GLenum >( shaderType );
    GLuint id = glCreateShaderProgramv( type, 1, &sourceCode );

    if( infoLog != nullptr )
    {
        *infoLog = getInfoLog();
    }

    GLint status;
    glGetProgramiv( id, GL_LINK_STATUS, &status );
    if( status == GL_TRUE )
    {
        m_id = id;
        m_type = shaderType;
    }
    else
    {
        glDeleteProgram( id );
    }
}

GLSeparableProgram::GLSeparableProgram( GLSeparableProgram&& move )
{
    destroy();
    m_id = move.m_id;
    m_type = move.m_type;
    move.m_id = 0;
    move.m_type = Type::NO_TYPE;
}

GLSeparableProgram& GLSeparableProgram::operator = (
    GLSeparableProgram&& move )
{
    if( this != &move )
    {
        destroy();
        m_id = move.m_id;
        m_type = move.m_type;
        move.m_id = 0;
        move.m_type = Type::NO_TYPE;
    }
    return *this;
}

// virtual
GLSeparableProgram::~GLSeparableProgram()
{
    destroy();
}

GLuint GLSeparableProgram::id() const
{
    return m_id;
}

bool GLSeparableProgram::isValid() const
{
    return( id() != 0 && type() != Type::NO_TYPE );
}

GLSeparableProgram::Type GLSeparableProgram::type() const
{
    return m_type;
}

void GLSeparableProgram::setUniformHandle( GLint uniformLocation, GLuint64 h )
{
    assert( isValid() );
    glProgramUniformHandleui64ARB( id(), uniformLocation, h );
}

void GLSeparableProgram::setUniformHandleArray( GLint uniformLocation,
    Array1DView< GLuint64 > handles )
{
    assert( isValid() );
    assert( handles.packed() );

    if( handles.packed() )
    {
        glProgramUniformHandleui64vARB( id(), uniformLocation, handles.size(),
            handles );
    }
}

void GLSeparableProgram::setUniformInt( GLint uniformLocation, int x )
{
    assert( isValid() );
    glProgramUniform1i( id(), uniformLocation, x );
}

void GLSeparableProgram::setUniformFloat( GLint uniformLocation, float x )
{
    assert( isValid() );
    glProgramUniform1f( id(), uniformLocation, x );
}

void GLSeparableProgram::setUniformMatrix4f( GLint uniformLocation, const Matrix4f& matrix )
{
    assert( isValid() );
    glProgramUniformMatrix4fv( id(), uniformLocation, 1, false, matrix );

}

void GLSeparableProgram::setUniformVector2f( GLint uniformLocation, const Vector2f& v )
{
    assert( isValid() );
    glProgramUniform2fv( id(), uniformLocation, 1, v );
}

void GLSeparableProgram::setUniformVector2i( GLint uniformLocation, const Vector2i& v )
{
    assert( isValid() );
    glProgramUniform2iv( id(), uniformLocation, 1, v );
}

void GLSeparableProgram::setUniformVector3f( GLint uniformLocation, const Vector3f& v )
{
    assert( isValid() );
    glProgramUniform3fv( id(), uniformLocation, 1, v );
}

void GLSeparableProgram::setUniformVector3i( GLint uniformLocation, const Vector3i& v )
{
    assert( isValid() );
    glProgramUniform3iv( id(), uniformLocation, 1, v );
}

void GLSeparableProgram::setUniformVector4f( GLint uniformLocation, const Vector4f& v )
{
    assert( isValid() );
    glProgramUniform4fv( id(), uniformLocation, 1, v );
}

void GLSeparableProgram::setUniformVector4i( GLint uniformLocation, const Vector4i& v )
{
    assert( isValid() );
    glProgramUniform4iv( id(), uniformLocation, 1, v );
}

GLint GLSeparableProgram::uniformLocation( const GLchar* name ) const
{
    return glGetUniformLocation( id(), name );
}

void GLSeparableProgram::destroy()
{
    if( m_id != 0 )
    {
        glDeleteProgram( m_id );
        m_id = 0;
    }
}

std::string GLSeparableProgram::getInfoLog() const
{
    GLint length;
    glGetProgramiv( id(), GL_INFO_LOG_LENGTH, &length );
    if( length > 1 )
    {
        std::string log( length - 1, '\0' );
        char* bufferStart = &( log[ 0 ] );
        GLsizei tmp;
        glGetProgramInfoLog( id(), length, &tmp, bufferStart );
        return log;
    }
    else
    {
        return std::string();
    }
}
