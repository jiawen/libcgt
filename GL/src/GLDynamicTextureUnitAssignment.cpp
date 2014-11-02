#include "GLDynamicTextureUnitAssignment.h"

#include "GLProgram.h"
#include "GLSamplerObject.h"
#include "GLTexture.h"

GLDynamicTextureUnitAssignment::GLDynamicTextureUnitAssignment( std::shared_ptr< GLProgram > pProgram ) :
	m_pProgram( pProgram ),
	m_count( 0 )
{

}
	
void GLDynamicTextureUnitAssignment::assign( const char* samplerName,
											std::shared_ptr< GLTexture > pTexture )
{
	glActiveTexture( GL_TEXTURE0 + m_count );
	pTexture->bind();
	GLSamplerObject::unbind( m_count );
	m_pProgram->setUniformInt( samplerName, m_count );
	++m_count;
}

void GLDynamicTextureUnitAssignment::assign( const char* samplerName,
											std::shared_ptr< GLTexture > pTexture,
											std::shared_ptr< GLSamplerObject > pSamplerObject )											
{
	glActiveTexture( GL_TEXTURE0 + m_count );
	pTexture->bind();
	pSamplerObject->bind( m_count );	
	m_pProgram->setUniformInt( samplerName, m_count );
	++m_count;
}

void GLDynamicTextureUnitAssignment::reset()
{
	m_count = 0;
}
