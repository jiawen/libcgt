#pragma once

#include <memory>
#include <vector>

#include <GL/glew.h>

class GLSamplerObject;
class GLSeparableProgram;
class GLTexture;

// A static assignment of textures to texture units and sampler names.
// Designed to be created once and used repeatedly on the same set of
// textures <--> samplers.
//
// For quick one-off assignments or dynamic assignments (such as ping-pong
// buffers), use GLDynamicTextureUnitAssignment.
class GLStaticTextureUnitAssignment
{
public:

    GLStaticTextureUnitAssignment(
        std::shared_ptr< GLSeparableProgram > program );

    // Assigns the texture with its default sampler object to a named sampler
    // in the linked program.
    //
    // A name should only be assigned exactly once.
    void assign( const char* samplerName, GLTexture* texture );

    // Assigns the texture and a sampler object to a named sampler in the
    // linked program.
    //
    // A name should only be assigned exactly once.
    void assign( const char* samplerName,
        GLTexture* texture, GLSamplerObject* samplerObject );

    // Resets assignments.
    void reset();

    // Iterates over assigned textures, binds them to their corresponding
    // texture units, and sets them in the program.
    void apply();

private:

    // TODO(jiawen): consider using raw pointers.
    std::shared_ptr< GLSeparableProgram > m_program;
    std::vector< GLTexture* > m_textures;
    std::vector< GLSamplerObject* > m_samplerObjects;
    std::vector< GLint > m_samplerLocations;

};
