#pragma once

#include "GLImageInternalFormat.h"

#include <vecmath/Vector2i.h>

class GLRenderbufferObject
{
public:

    // TODO: multisample: can pass samples = 0 to get exactly the normal
    // and check for GL_MAX_SAMPLES
    GLRenderbufferObject( const Vector2i& size, GLImageInternalFormat internalFormat );

    virtual ~GLRenderbufferObject();

    GLuint id() const;

private:

    GLuint m_id;

    Vector2i m_size;
    GLImageInternalFormat m_internalFormat;
};