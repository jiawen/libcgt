#ifndef GL_SHARED_H
#define GL_SHARED_H

#if 0

// singleton class for sharing simple OpenGL objects
#include <memory>

#include <GL/glew.h>
#include <QPair>
#include <QHash>
#include <QVector>

class GLBufferObject;
class GLFramebufferObject;
class GLTextureRectangle;

class GLShared
{
public:

    static std::shared_ptr< GLShared > getInstance();
    virtual ~GLShared();

    std::shared_ptr< GLFramebufferObject > getSharedFramebufferObject();

    // request count textures of size width by height
    QVector< GLTextureRectangle* > getSharedTexture( int width, int height, int count );

    // request a buffer object of size width by height
    GLBufferObject* getSharedXYCoordinateVBO( int width, int height );

private:

    GLShared();

    static std::shared_ptr< GLShared > s_singleton;

    std::shared_ptr< GLFramebufferObject > m_fbo;

    QHash< QPair< int, int >, QVector< GLTextureRectangle* > > m_qhSharedTextures;
    QHash< QPair< int, int >, GLBufferObject* > m_qhSharedXYVBOs;
};

#endif

#endif