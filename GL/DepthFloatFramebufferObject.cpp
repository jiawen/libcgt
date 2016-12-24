#include "DepthFloatFramebufferObject.h"

#include "libcgt/core/vecmath/Vector2i.h"

DepthFloatFramebufferObject::DepthFloatFramebufferObject( const Vector2i& size ) :
    m_depthFloat( size, GLImageInternalFormat::R32F ),
    m_depthStencil( size, GLImageInternalFormat::DEPTH24_STENCIL8 )
{
    m_fbo.attachTexture( GL_COLOR_ATTACHMENT0, &m_depthFloat );
    m_fbo.attachRenderbuffer( GL_DEPTH_ATTACHMENT, &m_depthStencil );
}
