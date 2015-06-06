#pragma once

class Vector2i;

#include "GLFramebufferObject.h"
#include "GLRenderbufferObject.h"
#include "GLTexture2D.h"

class DepthFloatFramebufferObject
{
public:

    DepthFloatFramebufferObject( const Vector2i& size );

    GLFramebufferObject m_fbo;
    GLTexture2D  m_depthFloat;
    GLRenderbufferObject m_depthStencil;

};
