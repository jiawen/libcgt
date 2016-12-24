#include "StandardFramebufferObject.h"

#include "libcgt/core/vecmath/Vector2i.h"

StandardFramebufferObject::StandardFramebufferObject( const Vector2i& size,
    GLImageInternalFormat colorFormat, GLImageInternalFormat depthFormat ) :
    m_color( size, colorFormat ),
    m_depthStencil( size, depthFormat )
{
    m_fbo.attachTexture( GL_COLOR_ATTACHMENT0, &m_color );
    m_fbo.attachRenderbuffer( GL_DEPTH_ATTACHMENT, &m_depthStencil );
}
