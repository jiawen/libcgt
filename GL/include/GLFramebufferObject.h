#pragma once

#include <GL/glew.h>

#include <common/Array1D.h>
#include <common/Array1DView.h>

class GLTexture;
class GLRenderbufferObject;

class GLFramebufferObject
{
public:

    // Get the id of the currently bound FBO, which may not correspond to a
    // libcgt GLFramebufferObject instance (such as from an external library).
    static GLuint boundId();

    // Bind an FBO allocated by an external library.
    static void bindExternalFBO( int id );

    // Unbind the one and only FBO
    // (and binds the default FBO, which has id 0).
	static void unbindAll();

    // The maximum number of color attachments.
	static GLint maxColorAttachments();

    // The maximum number of draw buffers (multiple render targets).
    static GLint maxNumDrawBuffers();

	GLFramebufferObject();
	virtual ~GLFramebufferObject();

    GLuint id() const;
	void bind();

	// TODO: GLTexture2D extends GLTexture
	// GLTexture->getType()
	// TODO: mipmaps


    // TODO(ARB_DSA): [EXT not available?] glClearNamedFramebuffer[fi|fv|iv|uiv]()
    // TODO: invalidate, blit

	// Attachment can be GL_COLOR_ATTACHMENT0, ... GL_COLOR_ATTACHMENTn,
	//   GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT.
    // layerIndex:
    //   If pTexture is a GLTexture2DArray, then layerIndex corresponds to the
    //     array index. (TODO: NOT SUPPORTED)
    //   If pTexture is a GLTexture3D
    // TODO: GL_TEXTURE_2D_MULTISAMPLE
	void attachTexture( GLenum attachment, GLTexture* pTexture, int mipmapLevel = 0,
        int layerIndex = 0 );
    void attachRenderbuffer( GLenum attachment, GLRenderbufferObject* pRenderbuffer );
	void detach( GLenum attachment );

    // Gets the framebuffer's index-ith draw buffer.
    // By default, drawbuffer i is assigned to GL_COLOR_ATTACHMENTi.
    GLenum getDrawBuffer( int index ) const;

    // Gets all maxNumDrawBuffers() draw buffers.
    // (The ones that are not set are GL_NONE).
    Array1D< GLenum > getDrawBuffers() const;

    // Set the framebuffer's 0th draw buffer to attachment.
    // By default, drawbuffer 0 is assigned to GL_COLOR_ATTACHMENT0.
    void setDrawBuffer( GLenum attachment );

    // Set the fraembuffer's draw buffers [0, 1, ... attachments.size())
    // to [attachments[0], attachments[1], ...)
    // attachments must be packed.
    void setDrawBuffers( Array1DView< GLenum > attachments );

    GLenum getReadBuffer() const;
    void setReadBuffer( GLenum attachment );

    // Returns the id of the object attached to attachment.
    // If the type is GL_RENDERBUFFER, then it's the renderbuffer object id.
    // If the type is GL_TEXTURE, then it's the texture id.
	GLuint getAttachedId( GLenum attachment );

    // Returns the type of the object attached to attachment.
    // It is one of:
    // GL_NONE, GL_FRAMEBUFFER_DEFAULT, GL_TEXTURE, or GL_RENDERBUFFER.
	GLuint getAttachedType( GLenum attachment );

    // Assuming the attached object is a texture, returns the mipmap level that
    // was attached.
    GLint getAttachedTextureMipMapLevel( GLenum attachment );

    // Assuming the attached object is a texture, returns the face index that
    // was attached. // TODO: check if it's GL_TEXTURE_CUBE_MAP_POSITIVE_X, etc, or 0, 1, 2...
    GLint getAttachedTextureCubeMapFace( GLenum attachment );    

    // TODO: query for red_size, ..., depth_size, stencil_size, etc
    // TODO: query for component_type, color_encoding

	bool checkStatus( GLenum* pStatus = nullptr );

private:	

	GLuint m_id;
};
