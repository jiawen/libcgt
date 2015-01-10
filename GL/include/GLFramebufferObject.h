#pragma once

#include <GL/glew.h>
#include <cstdio>

#include <common/BasicTypes.h>

class GLTexture;

class GLFramebufferObject
{
public:

    // Get the id of the currently bound FBO.
    static GLuint boundId();

    // Unbind the one and only FBO
    // (and binds the default FBO, which has id 0).
	static void unbindAll();
	static GLint getMaxColorAttachments();

	GLFramebufferObject();
	virtual ~GLFramebufferObject();

    GLuint id() const;
	void bind();

	// TODO: GLTexture2D extends GLTexture
	// GLTexture->getType()
	// TODO: mipmaps
    // TODO: glDrawBuffers:
    // glFramebufferDrawBufferEXT() / glFramebufferReadBufferEXT() / ...
    //    clear, invalidate, blit
    // ARB_DSA: glNamedFramebufferDrawBuffer() / glClearNamedFramebuffer()

	// Attachment can be GL_COLOR_ATTACHMENT0, ... GL_COLOR_ATTACHMENTn,
	// GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT.
	void attachTexture( GLenum attachment, GLTexture* pTexture, GLint zSlice = 0 );
	void detachTexture( GLenum attachment );

	// TODO: attachRenderBuffer()

	GLuint getAttachedId( GLenum attachment );
	GLuint getAttachedType( GLenum attachment );
	GLint getAttachedZSlice( GLenum attachment );

	bool checkStatus( GLenum* pStatus = NULL );

private:	

	GLuint m_id;
};
