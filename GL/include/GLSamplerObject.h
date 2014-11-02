#pragma once

#include <GL/glew.h>

class GLSamplerObject
{
public:

    // TODO: use this
    // TODO: mipmaps, anisotropic
    enum class GLTextureFilterMode
    {
        GLTextureFilterMode_NEAREST = GL_NEAREST,
        GLTextureFilterMode_LINEAR = GL_LINEAR
    };

    // TODO: enum class for wrap modes

	static GLfloat getLargestSupportedAnisotropy();
	static void unbind( GLuint textureUnit );

	GLSamplerObject();
	virtual ~GLSamplerObject();

	GLuint id() const;

	void bind( GLuint textureUnit );

	// filter modes
	GLint minFilterMode() const;
	GLint magFilterMode() const;

	void setMinFilterMode( GLint mode );
	void setMagFilterMode( GLint mode );
	// anisotropy \in [1, getLargestSupportedAnisotropy()]
	void setAnisotropy( GLfloat anisotropy );

	void setAllFiltersNearest();
	void setAllFiltersLinear();

	// wrap modes
	GLint wrapModeS() const;
	GLint wrapModeT() const;
	GLint wrapModeR() const;

	// TODO: set individual wrap modes...

	void setAllWrapModesClampToEdge();

private:

	GLuint m_id;
};
