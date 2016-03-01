#pragma once

#ifdef GL_PLATFORM_ES_31
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#endif
#ifdef GL_PLATFORM_45
#include <GL/glew.h>
#endif

enum class GLTextureFilterMode
{
    NEAREST = GL_NEAREST,
    LINEAR = GL_LINEAR,
    NEAREST_MIPMAP_NEAREST = GL_NEAREST_MIPMAP_NEAREST,
    NEAREST_MIPMAP_LINEAR = GL_NEAREST_MIPMAP_LINEAR,
    LINEAR_MIPMAP_NEAREST = GL_LINEAR_MIPMAP_NEAREST,
    LINEAR_MIPMAP_LINEAR = GL_LINEAR_MIPMAP_LINEAR
};

enum class GLWrapMode
{
    CLAMP_TO_EDGE = GL_CLAMP_TO_EDGE,
    REPEAT = GL_REPEAT,
#ifdef GL_PLATFORM_45
    MIRROR_CLAMP_TO_EDGE = GL_MIRROR_CLAMP_TO_EDGE,
#endif
    MIRRORED_REPEAT = GL_MIRRORED_REPEAT,

#ifdef GL_PLATFORM_45
    // You probably don't want this.
    CLAMP_TO_BORDER = GL_CLAMP_TO_BORDER
#endif
};

class GLSamplerObject
{
public:

    static GLfloat getLargestSupportedAnisotropy();
    // textureUnitIndex is an unsigned integer index (0, 1, ...)
    // and *not* a GLenum (GL_TEXTURE0, GL_TEXTURE1, ...)
    static void unbind( GLuint textureUnitIndex );

    GLSamplerObject();
    virtual ~GLSamplerObject();

    GLuint id() const;

    // textureUnitIndex is an unsigned integer index (0, 1, ...)
    // and *not* a GLenum (GL_TEXTURE0, GL_TEXTURE1, ...)
    void bind( GLuint textureUnitIndex );

    // Filter modes.
    GLTextureFilterMode minFilterMode() const;
    GLTextureFilterMode magFilterMode() const;
    GLfloat anisotropy() const;

    void setMinFilterMode( GLTextureFilterMode mode );
    void setMagFilterMode( GLTextureFilterMode mode );
    void setMinMagFilterMode( GLTextureFilterMode minMode, GLTextureFilterMode magMode );

    // Enable anisotropic filtering.
    // anisotropy is a value \in [1, getLargestSupportedAnisotropy()].
    // Setting it to 1 turns off anisotropic filtering.
    // Turning it on lets GL take more samples.
    void setAnisotropy( GLfloat anisotropy );

    // Wrap modes.
    // Note: OpenGL's texture coordinates are named (s, t, r). And rarely q.
    // GLSL, to avoid the overloaded r, calls them (s, t, p, q).
    GLWrapMode wrapModeS() const;
    GLWrapMode wrapModeT() const;
    GLWrapMode wrapModeR() const;

    // Set individual wrap modes.
    void setSWrapMode( GLWrapMode mode );
    void setTWrapMode( GLWrapMode mode );
    void setRWrapMode( GLWrapMode mode );
    // Set just s and t.
    void setSTWrapModes( GLWrapMode sMode, GLWrapMode tMode );
    void setSTWrapModes( GLWrapMode mode );
    // Set s, t, and r.
    void setWrapModes( GLWrapMode sMode, GLWrapMode tMode, GLWrapMode rMode );
    void setWrapModes( GLWrapMode mode );

private:

    GLuint m_id;
};
