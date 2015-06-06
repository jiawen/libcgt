#include "DepthFloatFramebufferObject.h"

#include <vecmath/Vector2i.h>

#include "GLFramebufferObject.h"
#include "GLTexture2D.h"
#include "GLRenderbufferObject.h"

DepthFloatFramebufferObject::DepthFloatFramebufferObject( const Vector2i& size )
{
    m_pFBO = std::make_shared< GLFramebufferObject >();
    m_pDepthFloat = std::make_shared< GLTexture2D >(
        size, GLImageInternalFormat::R32F );
    m_pDepthStencil = std::make_shared< GLRenderbufferObject >(
        size, GLImageInternalFormat::DEPTH24_STENCIL8 );

    m_pFBO->attachTexture( GL_COLOR_ATTACHMENT0, m_pDepthFloat.get() );
    m_pFBO->attachRenderbuffer( GL_DEPTH_ATTACHMENT, m_pDepthStencil.get() );
}