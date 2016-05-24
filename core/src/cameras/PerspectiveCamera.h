#pragma once

#include <string>

#include "Camera.h"
#include <math/MathUtils.h>

class PerspectiveCamera : public Camera
{
public:

    // camera at origin
    // x points right, y points up, z towards viewer
    // 90 degree field of view, 1:1 aspect ratio
    static const PerspectiveCamera CANONICAL;

    // camera at (0,0,5), looking at origin
    // x points right, y points up, z towards viewer
    // 50 degree field of view, 1:1 aspect ratio
    static const PerspectiveCamera FRONT;

    // camera at (5,0,0), looking at origin
    // x points into the screen, y points up, z points right
    // 50 degree field of view, 1:1 aspect ratio
    static const PerspectiveCamera RIGHT;

    // camera at (0,5,0), looking at origin
    // x points riht, y points into the screen, z points up
    // 50 degree field of view, 1:1 aspect ratio
    static const PerspectiveCamera TOP;

    // TODO: make isDirectX a template parameter

    // fovY: field of view angle in the y direction, in degrees
    // aspect: aspect ratio in width over height
    PerspectiveCamera( const Vector3f& eye = Vector3f( 0, 0, 5 ),
        const Vector3f& center = Vector3f( 0, 0, 0 ),
        const Vector3f& up = Vector3f( 0, 1, 0 ),
        float fovYRadians = libcgt::core::math::degreesToRadians( 50.0f ),
        float aspect = 1.0f,
        float zNear = 1.0f, float zFar = 100.0f,
        bool isDirectX = true );

    // gets the parameters used to set this perspective camera
    // note that these are simply the cached values
    // the state can become *inconsistent* if GLCamera::setFrustum()
    // calls are made
    //
    // TODO: maybe a PerspectiveCamera should disallow setting of top, right, etc
    // a skewed perspective camera for depth of field should return a general camera...
    void getPerspective( float& fovYRadians, float aspect,
        float& zNear, float& zFar );

    void setPerspective(
        float fovYRadians = libcgt::core::math::degreesToRadians( 50.0f ),
        float aspect = 1.0f,
        float zNear = 1.0f, float zFar = 100.0f );

    // same as below, but uses existing zNear and zFar
    void setPerspectiveFromIntrinsics( const Vector2f& focalLengthPixels,
        const Vector2f& imageSize );

    // Sets the perspective projection given computer-vision style intrinsics:
    // focal length and image size in pixels
    // Does not allow for radial distortion
    void setPerspectiveFromIntrinsics( const Vector2f& focalLengthPixels,
        const Vector2f& imageSize,
        float zNear, float zFar );

    // get/set the aspect ratio of the screen,
    // defined as width / height
    float aspect() const;
    void setAspect( float aspect );
    void setAspect( const Vector2f& screenSize );

    // get/set the field of view, in radians
    float fovXRadians() const;
    float fovYRadians() const;
    void setFovYRadians( float fovY );

    // get/set the field of view, in degrees
    float fovXDegrees() const;
    float fovYDegrees() const;
    void setFovYDegrees( float fovY );

    // returns half the field of view, in radians
    // very useful in projection math
    float halfFovXRadians() const;
    float halfFovYRadians() const;

    // returns tangent of half the field of view
    float tanHalfFovX() const;
    float tanHalfFovY() const;

    // When you change the near plane on a PerspectiveCamera, it also changes
    // left, right, bottom, and top of the frustum.
    virtual void setZNear( float zNear ) override;

    // for an image of size (width x height)
    // returns the "focal length in pixels"
    // i.e., the (positive) z such that the frustum
    // maps from (-width/2, -height/2) --> (width/2, height/2) in world units
    //
    // calculated as:
    // tan( halfFovX ) = (width/2) / focalLengthPixelsX
    Vector2f focalLengthPixels( const Vector2f& screenSize ) const;

    // Returns the OpenGL / Direct3D style projection matrix,
    // mapping eye space to clip space.
    virtual Matrix4f projectionMatrix() const override;

    // returns the projection matrix P such that
    // the plane at a distance focusZ in front of the center of the lens
    // is kept constant while the eye has been moved
    // by (eyeX, eyeY) *in the plane of the lens*
    // i.e. eyeX is in the direction of right()
    // and eyeY is in the direction of up()

    // TODO: jittered orthographic projection?
    // OpenGL / DirectX probably doesn't allow a parallelopiped as a viewing volume
    // could also just shift everything over
    Matrix4f jitteredProjectionMatrix( float eyeX, float eyeY, float focusZ ) const;

    // equivalent to jitteredProjectionMatrix() * jitteredViewMatrix()
    Matrix4f jitteredViewProjectionMatrix( float eyeX, float eyeY, float focusZ ) const;

    // TODO: harmonize this with math, maybe just make these static functions
    // for a perspective camera, you only need the field of view and viewport aspect ratio
    // for a skewed camera, you need left/right/bottom/top, along with zNear
    // (or any focal plane in general that's aligned with the view direction,
    // but the l/r/b/t need to be defined on that plane)
    //
    // TODO: harmonize the aspect ratio: the viewport does not have to be the same aspect ratio as that of the camera

    // given a pixel (x,y) in screen space (in [0,screenSize.x), [0,screenSize.y))
    // and an actual depth value (\in [0, +inf)), not distance along ray,
    // returns its eye space coordinates (right handed, output z will be negative), w = 1
    virtual Vector4f screenToEye( const Vector2i& xy, float depth, const Vector2f& screenSize ) const override;
    virtual Vector4f screenToEye( const Vector2f& xy, float depth, const Vector2f& screenSize ) const override;

    virtual std::string toString() const override;

    // Returns the 24 points corresponding to the 12 lines of a perspective frustum.
    // The first 8 points correspond to lines from the eye to each corner of the far plane.
    // The next 8 points correspond to lines of the near plane.
    // The last 8 points correspond to lines of the far plane.
    std::vector< Vector4f > frustumLines() const;

    // TODO: implement save using toString().
    // TODO: implement load using fromString().
    bool loadTXT( const std::string& filename );
    bool saveTXT( const std::string& filename );

    static PerspectiveCamera lerp( const PerspectiveCamera& c0, const PerspectiveCamera& c1, float t );
    static PerspectiveCamera cubicInterpolate( const PerspectiveCamera& c0, const PerspectiveCamera& c1, const PerspectiveCamera& c2, const PerspectiveCamera& c3, float t );

    // Copy the perpspective projection (aka intrinsics without a screen size), from --> to:
    // Copies fov, aspect ratio, zNear, and zFar.
    static void copyPerspective( const PerspectiveCamera& from, PerspectiveCamera& to );

private:

    // Updates the internal frustum values from m_fovYRadiand and m_aspect.
    // Called whenever these values change.
    void updateFrustum();

    float m_fovYRadians;
    float m_aspect;

};
