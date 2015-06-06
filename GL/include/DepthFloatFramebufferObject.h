#pragma once

#include <memory>

class Vector2i;
class GLFramebufferObject;
class GLTexture2D;
class GLRenderbufferObject;

class DepthFloatFramebufferObject
{
public:

    DepthFloatFramebufferObject( const Vector2i& size );

    std::shared_ptr< GLFramebufferObject > m_pFBO;
    std::shared_ptr< GLTexture2D > m_pDepthFloat;
    std::shared_ptr< GLRenderbufferObject > m_pDepthStencil;

};