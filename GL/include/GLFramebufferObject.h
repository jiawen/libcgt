#ifndef GL_FRAMEBUFFER_OBJECT_H
#define GL_FRAMEBUFFER_OBJECT_H

#include <GL/glew.h>
#include <cstdio>

#include <common/BasicTypes.h>

class GLTexture;

class GLFramebufferObject
{
public:

	static void unbind();
	static GLint getMaxColorAttachments();

	GLFramebufferObject();
	virtual ~GLFramebufferObject();

	void bind();

	// TODO: GLTexture2D extends GLTexture
	// GLTexture->getType()
	// forget mipmapped textures for a while

	// attachment can be GL_COLOR_ATTACHMENT0, ... GL_COLOR_ATTACHMENTn,
	// GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT
	void attachTexture( GLenum attachment, GLTexture* pTexture, GLint zSlice = 0 );
	void detachTexture( GLenum attachment );

	// TODO: attachRenderBuffer()

	GLuint getAttachedId( GLenum attachment );
	GLuint getAttachedType( GLenum attachment );
	GLint getAttachedZSlice( GLenum attachment );

	bool checkStatus( GLenum* pStatus = NULL );

private:	

	GLuint generateFBOId();

	GLuint m_id;
};

#endif
