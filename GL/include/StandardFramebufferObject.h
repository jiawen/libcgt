#pragma once

#include <memory>

#include "GLImageInternalFormat.h"

class Vector2i;
class GLFramebufferObject;
class GLTexture2D;
class GLRenderbufferObject;

class StandardFramebufferObject
{
public:

    StandardFramebufferObject( const Vector2i& size,
        GLImageInternalFormat colorFormat = GLImageInternalFormat::RGBA8,
        GLImageInternalFormat depthFormat = GLImageInternalFormat::DEPTH24_STENCIL8 );

    std::shared_ptr< GLFramebufferObject > m_pFBO;
    std::shared_ptr< GLTexture2D > m_pColor;
    std::shared_ptr< GLRenderbufferObject > m_pDepthStencil;

};