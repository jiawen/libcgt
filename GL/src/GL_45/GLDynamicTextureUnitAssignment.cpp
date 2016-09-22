#include "GLDynamicTextureUnitAssignment.h"

#include "GLSamplerObject.h"
#include "GLSeparableProgram.h"
#include "GLTexture.h"

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
