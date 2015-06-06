#include "StandardFramebufferObject.h"

#include <vecmath/Vector2i.h>

#include "GLFramebufferObject.h"
#include "GLTexture2D.h"
#include "GLRenderbufferObject.h"

StandardFramebufferObject::StandardFramebufferObject( const Vector2i& size,
    GLImageInternalFormat colorFormat, GLImageInternalFormat depthFormat )
{
    m_pFBO = std::make_shared< GLFramebufferObject >();
    m_pColor = std::make_shared< GLTexture2D >(
        size, colorFormat );
    m_pDepthStencil = std::make_shared< GLRenderbufferObject >(
        size, depthFormat );

    m_pFBO->attachTexture( GL_COLOR_ATTACHMENT0, m_pColor.get() );
    m_pFBO->attachRenderbuffer( GL_DEPTH_ATTACHMENT, m_pDepthStencil.get() );
}