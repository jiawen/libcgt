#include <cstdio>

#include "libcgt/core/common/ArrayUtils.h"
#include "libcgt/core/io/PNGIO.h"
#include "libcgt/core/io/PortableFloatMapIO.h"

#ifdef GL_PLATFORM_ES_31
#include "GLES_31/GLTexture2D.h"
#endif
#ifdef GL_PLATFORM_45
#include "GL_45/GLTexture2D.h"
#endif
#include "GLUtilities.h"

using libcgt::core::arrayutils::componentView;

// static
void GLUtilities::clearDepth( GLfloat d )
{
    glClearDepthf( d );
}

#ifdef GL_PLATFORM_45
// static
void GLUtilities::clearDepth( GLdouble d )
{
    glClearDepth( d );
}

// static
void GLUtilities::clearDepthNV( GLdouble d )
{
    glClearDepthdNV( d );
}
#endif

// static
Rect2i GLUtilities::getViewport()
{
    int vp[4];
    glGetIntegerv( GL_VIEWPORT, vp );
    return{ { vp[0], vp[1] }, { vp[2], vp[3] } };
}

// static
Box3f GLUtilities::getViewport3D()
{
    float vp[4];
    glGetFloatv( GL_VIEWPORT, vp );

    float dr[2];
    glGetFloatv( GL_DEPTH_RANGE, dr );

    float zNear = dr[0];
    float zFar = dr[1];
    return{ { vp[0], vp[1], zNear }, { vp[2], vp[3], zFar - zNear } };
}

// static
void GLUtilities::setViewport( const Rect2i& vp )
{
    glViewport( vp.origin.x, vp.origin.y, vp.size.x, vp.size.y );
}

// static
void GLUtilities::setDepthRange( GLfloat zNear, GLfloat zFar )
{
    glDepthRangef( zNear, zFar );
}

#ifdef GL_PLATFORM_45
// static
void GLUtilities::setDepthRange( GLdouble zNear, GLdouble zFar )
{
    glDepthRange( zNear, zFar );
}

// static
void GLUtilities::setDepthRangeNV( double zNear, double zFar )
{
    glDepthRangedNV( zNear, zFar );
}
#endif

// static
void GLUtilities::printGLRenderer()
{
    printf( "%s\n", glGetString( GL_RENDERER ) );
}

// static
void GLUtilities::printGLVendor()
{
    printf( "%s\n", glGetString( GL_VENDOR ) );
}

// static
void GLUtilities::printGLVersion()
{
    printf( "%s\n", glGetString( GL_VERSION ) );
}

// static
void GLUtilities::printLastError()
{
    fprintf( stderr, "%s\n", getLastErrorString().c_str() );
}

// static
std::string GLUtilities::getLastErrorString()
{
    GLenum glError = glGetError();
    switch( glError )
    {
    case GL_NO_ERROR:
        return "GL_NO_ERROR";
    case GL_INVALID_ENUM:
        return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE:
        return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION:
        return "GL_INVALID_OPERATION";
    case GL_INVALID_FRAMEBUFFER_OPERATION:
        return "GL_INVALID_FRAMEBUFFER_OPERATION";
    case GL_OUT_OF_MEMORY:
        return "GL_OUT_OF_MEMORY";
#ifdef GL_PLATFORM_45
    case GL_STACK_UNDERFLOW:
        return "GL_STACK_UNDERFLOW";
    case GL_STACK_OVERFLOW:
        return "GL_STACK_OVERFLOW";
#endif
    default:
        // TODO: to_string hex
        return "Unknown error: " +
            std::to_string( static_cast< unsigned int >( glError ) );
    }
}
