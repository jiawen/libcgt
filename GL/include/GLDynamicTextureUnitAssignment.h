#pragma once

#include <memory>

#include <GL/glew.h>

class GLProgram;
class GLSamplerObject;
class GLTexture;

// A dynamic assignment of textures to texture units and sampler names.
// Designed to be created every rendering cycle and discarded.
//
// For a static, reusable assignment, use GLStaticTextureUnitAssignment.
class GLDynamicTextureUnitAssignment
{
public:

	GLDynamicTextureUnitAssignment( std::shared_ptr< GLProgram > pProgram );

	// Binds pTexture to the next available texture unit i (starting at 0),
	// and sets the sampler to i.
	void assign( const char* samplerName, std::shared_ptr< GLTexture > pTexture );

	// Binds pTexture and pSamplerObject to the next available texture unit i
	// (starting at 0), and sets the sampler to i.
	void assign( const char* samplerName,
		std::shared_ptr< GLTexture > pTexture, std::shared_ptr< GLSamplerObject > pSamplerObject );

	// Resets the counter to 0.
	void reset();

private:

	std::shared_ptr< GLProgram > m_pProgram;
	int m_count;
};
