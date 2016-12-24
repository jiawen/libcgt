#pragma once

class Vector2i;

#ifdef GL_PLATFORM_ES_31
#include "../GLES_31/GLFramebufferObject.h"
#include "../GLES_31/GLRenderbufferObject.h"
#include "../GLES_31/GLTexture2D.h"
#endif
#ifdef GL_PLATFORM_45
#include "../GL_45/GLFramebufferObject.h"
#include "../GL_45/GLRenderbufferObject.h"
#include "../GL_45/GLTexture2D.h"
#endif

class DepthFloatFramebufferObject
{
public:

    DepthFloatFramebufferObject( const Vector2i& size );

    GLFramebufferObject m_fbo;
    GLTexture2D m_depthFloat;
    GLRenderbufferObject m_depthStencil;

};
