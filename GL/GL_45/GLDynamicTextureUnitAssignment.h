#pragma once

#include <memory>

class GLSamplerObject;
class GLSeparableProgram;
class GLTexture;

// A dynamic assignment of textures to texture units and sampler names.
// Designed to be created every rendering cycle and discarded.
//
// For a static, reusable assignment, use GLStaticTextureUnitAssignment.
class GLDynamicTextureUnitAssignment
{
public:

    GLDynamicTextureUnitAssignment(
        std::shared_ptr< GLSeparableProgram > program );

    // Binds pTexture to the next available texture unit i (starting at 0),
    // and sets the sampler to i.
    void assign( const char* samplerName, GLTexture& texture );

    // Binds pTexture and pSamplerObject to the next available texture unit i
    // (starting at 0), and sets the sampler to i.
    void assign( const char* samplerName,
        GLTexture& texture, GLSamplerObject& samplerObject );

    // Resets the counter to 0.
    void reset();

private:

    // TODO(jiawen): consider using a raw pointer.
    std::shared_ptr< GLSeparableProgram > m_program;
    int m_count;
};
