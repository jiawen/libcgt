#pragma once

class Vector2i;

#include "GLFramebufferObject.h"
#include "GLImageInternalFormat.h"
#include "GLRenderbufferObject.h"
#include "GLTexture2D.h"

class StandardFramebufferObject
{
public:

    StandardFramebufferObject( const Vector2i& size,
        GLImageInternalFormat colorFormat = GLImageInternalFormat::RGBA8,
        GLImageInternalFormat depthFormat = GLImageInternalFormat::DEPTH24_STENCIL8 );

    GLFramebufferObject m_fbo;
    GLTexture2D m_color;
    GLRenderbufferObject m_depthStencil;

};
