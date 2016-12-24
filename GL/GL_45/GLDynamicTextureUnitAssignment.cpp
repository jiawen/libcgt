#include "GLDynamicTextureUnitAssignment.h"

#include "libcgt/GL/GLSamplerObject.h"
#include "libcgt/GL/GLSeparableProgram.h"
#include "libcgt/GL/GL_45/GLTexture.h"

GLDynamicTextureUnitAssignment::GLDynamicTextureUnitAssignment(
    std::shared_ptr< GLSeparableProgram > program ) :
    m_program( program ),
    m_count( 0 )
{

}

void GLDynamicTextureUnitAssignment::assign( const char* samplerName,
    GLTexture& texture )
{
    texture.bind( m_count );
    GLSamplerObject::unbind( m_count );
    m_program->setUniformInt( m_program->uniformLocation( samplerName ),
        m_count );
    ++m_count;
}

void GLDynamicTextureUnitAssignment::assign( const char* samplerName,
    GLTexture& texture, GLSamplerObject& samplerObject )
{
    texture.bind( m_count );
    samplerObject.bind( m_count );
    m_program->setUniformInt( m_program->uniformLocation( samplerName ),
        m_count );
    ++m_count;
}

void GLDynamicTextureUnitAssignment::reset()
{
    m_count = 0;
}
