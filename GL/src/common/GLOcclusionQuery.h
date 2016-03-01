#ifndef GL_OCCLUSION_QUERY_H
#define GL_OCCLUSION_QUERY_H

#ifdef GL_PLATFORM_ES_31
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#endif
#ifdef GL_PLATFORM_45
#include <GL/glew.h>
#endif

class GLOcclusionQuery
{
public:

    // TODO: modernize this API. There are different types of queries.
#ifdef GL_PLATFORM_45
    static GLuint getCurrentQuery();
    static GLint nBits();
#endif

    GLOcclusionQuery();
    virtual ~GLOcclusionQuery();
    GLuint getQueryId();

#ifdef GL_PLATFORM_45
    void begin();
    void end();
#endif

    bool isResultAvailable();
    GLuint getResult();

private:

    GLuint m_uiQueryId;

};

#endif
