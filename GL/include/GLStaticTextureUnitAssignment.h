#pragma once

#include <memory>
#include <vector>

#include <GL/glew.h>

class GLProgram;
class GLSamplerObject;
class GLTexture;

// A static assignment of textures to texture units and sampler names.
// Designed to be created once and used repeatedly on the same set of
// textures <--> samplers.
//
// For quick one-off assignments or dynamic assignments (such as ping-pong buffers),
// use GLDynamicTextureUnitAssignment.
class GLStaticTextureUnitAssignment
{
public:

	GLStaticTextureUnitAssignment( std::shared_ptr< GLProgram > pProgram );

	// Assigns the texture with its default sampler object
	// to a named sampler in the linked program.
	//
	// A name should only be assigned exactly once.
	void assign( const char* samplerName,
		std::shared_ptr< GLTexture > pTexture );

	// Assigns the texture and a sampler object
	// to a named sampler in the linked program.
	//
	// A name should only be assigned exactly once.
	void assign( const char* samplerName,
		std::shared_ptr< GLTexture > pTexture, std::shared_ptr< GLSamplerObject > pSamplerObject );

	// Resets assignments
	void reset();

	// Iterates over assigned textures, binds them
	// to their corresponding texture units,
	// and sets them in the program.
	void apply();

private:

	std::shared_ptr< GLProgram > m_pProgram;
	std::vector< std::shared_ptr< GLTexture > > m_textures;
	std::vector< std::shared_ptr< GLSamplerObject > > m_samplerObjects;
	std::vector< GLint > m_samplerLocations;

};
