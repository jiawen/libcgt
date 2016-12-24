#pragma once

#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>

// TODO: implement me.
class GLTexture2D;
class GLRenderbufferObject;

class GLFramebufferObject
{
public:

    void attachTexture( GLenum attachment, GLTexture2D* pTexture,
        int mipmapLevel = 0 )
    {

    }

    void attachRenderbuffer( GLenum attachment, GLRenderbufferObject* pRenderbuffer )
    {

    }
};
