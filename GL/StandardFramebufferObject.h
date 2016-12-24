#pragma once

class Vector2i;

#ifdef GL_PLATFORM_ES_31
#include "libcgt/GL/GLES_31/GLFramebufferObject.h"
#include "libcgt/GL/GLES_31/GLRenderbufferObject.h"
#include "libcgt/GL/GLES_31/GLTexture2D.h"
#endif
#ifdef GL_PLATFORM_45
#include "libcgt/GL/GL_45/GLFramebufferObject.h"
#include "libcgt/GL/GL_45/GLRenderbufferObject.h"
#include "libcgt/GL/GL_45/GLTexture2D.h"
#endif

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
