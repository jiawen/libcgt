#pragma once

#include <GL/glew.h>

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
    MIRROR_CLAMP_TO_EDGE = GL_MIRROR_CLAMP_TO_EDGE,
    MIRRORED_REPEAT = GL_MIRRORED_REPEAT,

    // You probably don't want this.
    CLAMP_TO_BORDER = GL_CLAMP_TO_BORDER
};

class GLSamplerObject
{
public:

	static GLfloat getLargestSupportedAnisotropy();
	static void unbind( GLuint textureUnit );

	GLSamplerObject();
	virtual ~GLSamplerObject();

	GLuint id() const;

	void bind( GLuint textureUnit );

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
