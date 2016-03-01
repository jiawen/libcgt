#include "GLStaticTextureUnitAssignment.h"

#include "GLProgram.h"
#include "GLSamplerObject.h"
#include "GLTexture.h"

GLStaticTextureUnitAssignment::GLStaticTextureUnitAssignment( std::shared_ptr< GLProgram > pProgram ) :
    m_pProgram( pProgram )
{

}

void GLStaticTextureUnitAssignment::assign( const char* samplerName,
                                           std::shared_ptr< GLTexture > pTexture )
{
    assign( samplerName, pTexture, nullptr );
}

void GLStaticTextureUnitAssignment::assign( const char* samplerName,
                                           std::shared_ptr< GLTexture > pTexture,
                                           std::shared_ptr< GLSamplerObject > pSamplerObject )
{
    m_textures.push_back( pTexture );
    m_samplerObjects.push_back( pSamplerObject );
    m_samplerLocations.push_back( m_pProgram->uniformLocation( samplerName ) );
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
        if( m_samplerObjects[i].get() != nullptr )
        {
            m_samplerObjects[i]->bind( i );
        }
        else
        {
            GLSamplerObject::unbind( i );
        }
        m_pProgram->setUniformInt( m_samplerLocations[i], i );
    }
}
