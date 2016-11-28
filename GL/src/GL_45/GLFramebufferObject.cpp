#include "GLFramebufferObject.h"

#include <cassert>
#include <cstdio>

#include "GLTexture2D.h"
#include "GLTexture3D.h"
#include "GLTextureCubeMap.h"
#include "GLTextureRectangle.h"
#include "GLRenderbufferObject.h"

// static
GLuint GLFramebufferObject::boundId()
{
    int bid;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING, &bid );
    return bid;
}

// static
void GLFramebufferObject::bindExternalFBO( int id )
{
    glBindFramebuffer( GL_FRAMEBUFFER, id );
}

// static
void GLFramebufferObject::unbindAll()
{
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}

// static
GLint GLFramebufferObject::maxColorAttachments()
{
    GLint maxColorAttachments;
    glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS, &maxColorAttachments );
    return maxColorAttachments;
}

// static
GLint GLFramebufferObject::maxNumDrawBuffers()
{
    GLint maxDrawBuffers;
    glGetIntegerv( GL_MAX_DRAW_BUFFERS, &maxDrawBuffers );
    return maxDrawBuffers;
}

GLFramebufferObject::GLFramebufferObject() :
    m_isExternal( false )
{
    glCreateFramebuffers( 1, &m_id );
}

GLFramebufferObject::GLFramebufferObject( int externalId ) :
    m_id( externalId ),
    m_isExternal( true )
{

}

GLFramebufferObject::GLFramebufferObject( GLFramebufferObject&& move )
{
    destroy();
    m_id = move.m_id;
    m_isExternal = move.m_isExternal;
    move.m_id = 0;
    move.m_isExternal = true;
}

GLFramebufferObject& GLFramebufferObject::operator = (
    GLFramebufferObject&& move )
{
    if( this != &move )
    {
        destroy();
        m_id = move.m_id;
        m_isExternal = move.m_isExternal;
        move.m_id = 0;
        move.m_isExternal = true;
    }
    return *this;
}

GLFramebufferObject::~GLFramebufferObject()
{
    destroy();
}

GLuint GLFramebufferObject::id() const
{
    return m_id;
}

bool GLFramebufferObject::isExternal() const
{
    return m_isExternal;
}

void GLFramebufferObject::bind()
{
    glBindFramebuffer( GL_FRAMEBUFFER, m_id );
}

void GLFramebufferObject::attachTexture( GLenum attachment, GLTexture2D* pTexture, int mipmapLevel )
{
    glNamedFramebufferTexture( m_id, attachment, pTexture->id(), mipmapLevel );
}

void GLFramebufferObject::attachTexture( GLenum attachment, GLTexture3D* pTexture, int zSlice, int mipmapLevel )
{
    glNamedFramebufferTextureLayer( m_id, attachment, pTexture->id(), mipmapLevel, zSlice );
}

void GLFramebufferObject::attachTexture( GLenum attachment, GLTextureRectangle* pTexture, int mipmapLevel )
{
    glNamedFramebufferTexture( m_id, attachment, pTexture->id(), mipmapLevel );
}

void GLFramebufferObject::attachTexture( GLenum attachment,
    GLTextureCubeMap* pTexture, GLCubeMapFace face, int mipmapLevel )
{
    glNamedFramebufferTextureLayer( m_id, attachment, pTexture->id(),
        mipmapLevel, static_cast< GLint >( face ) );
}

void GLFramebufferObject::detach( GLenum attachment )
{
    GLenum type = getAttachedType( attachment );
    switch( type )
    {
    case GL_NONE:
        break;

    case GL_FRAMEBUFFER_DEFAULT:
        break;

    case GL_RENDERBUFFER:
        // 0 ==> detach
        glNamedFramebufferRenderbuffer( m_id, attachment, GL_RENDERBUFFER, 0 );
        break;

    case GL_TEXTURE:
        glNamedFramebufferTexture( m_id, attachment,
            0, // texture id, 0 ==> detach
            0 ); // mipmap level
        break;

    default:
        fprintf( stderr, "GLFramebufferObject::detach() ERROR: Unknown attached resource type\n" );
        assert( false );
        break;
    }
}

void GLFramebufferObject::attachRenderbuffer( GLenum attachment, GLRenderbufferObject* pRenderbuffer )
{
    glNamedFramebufferRenderbuffer( m_id, attachment, GL_RENDERBUFFER, pRenderbuffer->id() );
}

GLuint GLFramebufferObject::getAttachedId( GLenum attachment )
{
    GLint id;
    glGetNamedFramebufferAttachmentParameteriv( m_id, attachment,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &id );
    return id;
}

GLuint GLFramebufferObject::getAttachedType( GLenum attachment )
{
    GLint type = 0;
    glGetNamedFramebufferAttachmentParameteriv( m_id, attachment,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &type );
    return type;
}

void GLFramebufferObject::setDrawBuffer( GLenum attachment )
{
    glNamedFramebufferDrawBuffer( m_id, attachment );
}

void GLFramebufferObject::setDrawBuffers(
    Array1DReadView< GLenum > attachments )
{
    assert( attachments.packed() );
    glNamedFramebufferDrawBuffers( m_id,
        static_cast< GLsizei >( attachments.size() ), attachments.pointer() );
}

void GLFramebufferObject::setReadBuffer( GLenum attachment )
{
    glFramebufferReadBufferEXT( m_id, attachment );
}

bool GLFramebufferObject::checkStatus( GLenum* pStatus )
{
    bool isComplete = false;

    GLenum status = glCheckNamedFramebufferStatus( m_id, GL_FRAMEBUFFER );
    switch( status )
    {
    case GL_FRAMEBUFFER_COMPLETE:
        isComplete = true;
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        fprintf( stderr, "Framebuffer incomplete: incomplete attachment.\n" );
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        fprintf( stderr, "Framebuffer incomplete: no attachments.\n" );
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
        fprintf( stderr,
            "Framebuffer incomplete: drawbuffer set to GL_NONE.\n" );
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
        fprintf( stderr,
            "framebuffer incpmplete: readbuffer set to GL_NONE.\n" );
        break;
    case GL_FRAMEBUFFER_UNSUPPORTED:
        fprintf( stderr, "Framebuffer incomplete: format unsupported.\n" );
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
        fprintf( stderr, "Framebuffer incomplete: inconsistent number of"
            " multisamples.\n" );
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
        fprintf( stderr, "Framebuffer incomplete: inconsistent layering.\n" );
    default:
        fprintf( stderr, "Can't get here!\n" );
        assert( false );
    }

    if( pStatus != nullptr )
    {
        *pStatus = status;
    }
    return isComplete;
}

void GLFramebufferObject::clearColor( int drawbufferIndex,
    const int8x4& color )
{
    glClearNamedFramebufferiv( m_id, GL_COLOR,
        GL_DRAW_BUFFER0 + drawbufferIndex,
        reinterpret_cast< const GLint* >( &color ) );
}

void GLFramebufferObject::clearColor( int drawbufferIndex,
    const uint8x4& color )
{
    glClearNamedFramebufferuiv( m_id, GL_COLOR, GL_DRAW_BUFFER0 + drawbufferIndex,
        reinterpret_cast< const GLuint* >( &color ) );
}

void GLFramebufferObject::clearColor( int drawbufferIndex, const Vector4f& color )
{
    Vector4f c = color;
    glClearNamedFramebufferfv( m_id, GL_COLOR, GL_DRAW_BUFFER0 + drawbufferIndex, c );
}

void GLFramebufferObject::clearDepth( int drawbufferIndex, float depth )
{
    glClearNamedFramebufferfv( m_id, GL_DEPTH, 0, &depth );
}

void GLFramebufferObject::clearStencil( int drawbufferIndex, int stencil )
{
    glClearNamedFramebufferiv( m_id, GL_STENCIL, drawbufferIndex, &stencil );
}

void GLFramebufferObject::clearDepthStencil( int drawbufferIndex, float depth, int stencil )
{
    glClearNamedFramebufferfi( m_id, GL_DEPTH_STENCIL, drawbufferIndex, depth, stencil );
}

void GLFramebufferObject::destroy()
{
    // Don't delete the default FBO if it was wrapped.
    if( !m_isExternal )
    {
        glDeleteFramebuffers( 1, &m_id );
        m_id = 0;
    }
}
