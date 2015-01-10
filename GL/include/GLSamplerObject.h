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

// TODO: use this
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

	// filter modes
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

	// wrap modes
	GLint wrapModeS() const;
	GLint wrapModeT() const;
	GLint wrapModeR() const;

	// TODO: set individual wrap modes...

	void setAllWrapModesClampToEdge();

private:

	GLuint m_id;
};
