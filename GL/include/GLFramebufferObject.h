#pragma once

#include <GL/glew.h>

#include <common/Array1D.h>
#include <common/Array1DView.h>
#include <common/BasicTypes.h>
#include <vecmath/Vector4f.h>

class GLTexture2D;
class GLTexture3D;
class GLTextureCubeMap;
enum class GLCubeMapFace;
class GLTextureRectangle;
class GLRenderbufferObject;

// TODO: default framebuffer / the one that was already given by Qt.
// Can sorta make one up: wrapFramebuffer( int externalId ). If it's external, don't delete it.
class GLFramebufferObject
{
public:

    // Get the id of the currently bound FBO, which may not correspond to a
    // libcgt GLFramebufferObject instance (such as from an external library).
    static GLuint boundId();

    // Bind an FBO allocated by an external library.
    // TODO: remove this and use GLFramebufferObjectStack.
    static void bindExternalFBO( int id );

    // Unbind the one and only FBO
    // (and binds the default FBO, which has id 0).
    static void unbindAll();

    // The maximum number of color attachments.
    static GLint maxColorAttachments();

    // The maximum number of draw buffers (multiple render targets).
    static GLint maxNumDrawBuffers();

    // Create a new blank framebuffer object.
    GLFramebufferObject();

    // Wraps a framebuffer object around an existing FBO id.
    // This is useful to make an object oriented interface around an externally
    // passed-in FBO, such as the default framebuffer, or the one created by
    // QOpenGLWidget.
    GLFramebufferObject( int externalId );

    virtual ~GLFramebufferObject();

    GLuint id() const;

    bool isExternal() const;

    void bind();

    // TODO: GLTexture2D extends GLTexture
    // GLTexture->getType()
    // TODO: mipmaps

    // TODO: ARB_DSA: invalidate, blit

    // Attachment can be GL_COLOR_ATTACHMENT0, ... GL_COLOR_ATTACHMENTn,
    //   GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT.
    void attachTexture( GLenum attachment, GLTexture2D* pTexture,
        int mipmapLevel = 0 );

    // For each mipmap level:
    //   Each z-slice of that level is treated as a layer.
    //   Note that each level can have a different z-size (along with x and y).
    void attachTexture( GLenum attachment, GLTexture3D* pTexture,
        int zSlice, int mipmapLevel = 0 );
    void attachTexture( GLenum attachment, GLTextureRectangle* pTexture,
        int mipmapLevel = 0 );
    // TODO: attachTextureArray( GLTexture2DArray: use
    //   glNamedFramebufferTextureLayer for a single layer index,
    //   use glNamedFramebufferTexture to attach them all at once ).
    // Note: layers are NOT attachments! You can have as many layers as you want.
    // The vertex shader can choose. You redirect an entire primitive at once
    // in a vertex or geometry shader to a layer.
    // TODO: learn about ARB_viewport_array.
    void attachTexture( GLenum attachment, GLTextureCubeMap* pTexture,
        GLCubeMapFace face, int mipmapLevel = 0 );
    // TODO(low_priority): attachTextureArray( cube map array )
    // TODO(super low priority): 1D texture works and is considered a 2D
    // FBO with height 1.
    // TODO(super low priority): 1D texture array: also uses layered
    // TODO: GL_TEXTURE_2D_MULTISAMPLE
    // Note: glNamedFramebufferTexture attaches all layers simultaneously.
    // Can in theory attach all z-slices of a cube map at once.
    // But that's a little crazy.

    void attachRenderbuffer( GLenum attachment, GLRenderbufferObject* pRenderbuffer );
    void detach( GLenum attachment );

    // Set the framebuffer's 0th draw buffer to attachment.
    // By default, drawbuffer 0 is assigned to GL_COLOR_ATTACHMENT0.
    void setDrawBuffer( GLenum attachment );

    // Set the fraembuffer's draw buffers [0, 1, ... attachments.size())
    // to [attachments[0], attachments[1], ...)
    // attachments must be packed.
    void setDrawBuffers( Array1DView< const GLenum > attachments );

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

    // This checks only the status of the draw buffer.
    bool checkStatus( GLenum* pStatus = nullptr );

    void clearColor( int drawBufferIndex, const int8x4& color );
    void clearColor( int drawBufferIndex, const uint8x4& color );
    void clearColor( int drawBufferIndex, const Vector4f& color );

    // Clear the depth buffer.
    // If depth is fixed point, it's clamped to [0,1] then converted to fixed point.
    // If depth is floating point, it is *still* clamped but no conversion is done.
    // You'll have to use glDepthRangedNV().
    void clearDepth( float depth );

    // Clear the stencil buffer.
    void clearStencil( int stencil );

    // Clear the depth and stencil attachments of this FBO simultaneously.
    // If depth is fixed point, it's clamped to [0,1] then converted to fixed point.
    // If depth is floating point, it is *still* clamped but no conversion is done.
    // You'll have to use glDepthRangedNV().
    void clearDepthStencil( float depth, int stencil );

private:

    GLuint m_id;
    bool m_isExternal;
};
