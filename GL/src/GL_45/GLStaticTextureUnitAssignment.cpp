#include "GLStaticTextureUnitAssignment.h"

#include "GLSamplerObject.h"
#include "GLSeparableProgram.h"
#include "GLTexture.h"

GLStaticTextureUnitAssignment::GLStaticTextureUnitAssignment(
    std::shared_ptr< GLSeparableProgram > program ) :
    m_program( program )
{

}

void GLStaticTextureUnitAssignment::assign( const char* samplerName,
    GLTexture* texture )
{
    assign( samplerName, texture, nullptr );
}

void GLStaticTextureUnitAssignment::assign( const char* samplerName,
    GLTexture* texture, GLSamplerObject* samplerObject )
{
    m_textures.push_back( texture );
    m_samplerObjects.push_back( samplerObject );
    m_samplerLocations.push_back( m_program->uniformLocation( samplerName ) );
}

void GLStaticTextureUnitAssignment::reset()
{
    m_textures.clear();
    m_samplerObjects.clear();
    m_samplerLocations.clear();
}

void GLStaticTextureUnitAssignment::apply()
{
    for( int i = 0; i < static_cast< int >( m_textures.size() ); ++i )
    {
        m_textures[i]->bind( i );
        if( m_samplerObjects[i] != nullptr )
        {
            m_samplerObjects[i]->bind( i );
        }
        else
        {
            GLSamplerObject::unbind( i );
        }
        m_program->setUniformInt( m_samplerLocations[i], i );
    }
}
