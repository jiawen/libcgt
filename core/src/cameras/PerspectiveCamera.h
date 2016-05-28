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

    // TODO(jiawen): provide more constructors.

    PerspectiveCamera(
        const libcgt::core::vecmath::EuclideanTransform& cameraFromWorld,
        const libcgt::core::cameras::GLFrustum& frustum,
        bool isDirectX = false );

    PerspectiveCamera(
        const Vector3f& eye, const Vector3f& center, const Vector3f& up,
        const libcgt::core::cameras::GLFrustum& frustum,
        bool isDirectX = false );

    PerspectiveCamera(
        const Vector3f& eye = Vector3f( 0, 0, 5 ),
        const Vector3f& center = Vector3f( 0, 0, 0 ),
        const Vector3f& up = Vector3f( 0, 1, 0 ),
        float fovYRadians = libcgt::core::math::degreesToRadians(50.0f),
        float aspectRatio = 1.0f,
        float zNear = 1.0f, float zFar = 100.0f,
        bool isDirectX = false );

    void setFrustum( const libcgt::core::cameras::GLFrustum& frustum );

    // Same as below, but uses existing zNear and zFar.
    void setFrustumFromIntrinsics(
        const libcgt::core::cameras::Intrinsics& intrinsics,
        const Vector2f& imageSize );

    // Sets this camera's GL-style frustum parameters from camera intrinsics.
    // focal length, principal point, and image size are in pixels.
    // The principal point has the y-axis pointing *up* on the image:
    // this is opposite the usual computer-vision convension.
    void setFrustumFromIntrinsics(
        const libcgt::core::cameras::Intrinsics& intrinsics,
        const Vector2f& imageSize, float zNear, float zFar );

    // Returns the OpenGL / Direct3D style projection matrix,
    // mapping eye space to clip space.
    virtual Matrix4f projectionMatrix() const override;

    // Returns the projection matrix P such that the plane at a distance focusZ
    // in front of the center of the lens is kept constant while the eye has
    // been moved by (eyeX, eyeY) *in the plane of the lens*.
    // I.e. eyeX is in the direction of right() and eyeY is in the direction of
    // up().
    Matrix4f jitteredProjectionMatrix( float eyeX, float eyeY, float focusZ ) const;

    // Equivalent to jitteredProjectionMatrix() * jitteredViewMatrix().
    Matrix4f jitteredViewProjectionMatrix( float eyeX, float eyeY, float focusZ ) const;

    // TODO(jiawen): implement fromString().
    virtual std::string toString() const override;

    // Returns the 24 points corresponding to the 12 lines of a perspective frustum.
    // The first 8 points correspond to lines from the eye to each corner of the far plane.
    // The next 8 points correspond to lines of the near plane.
    // The last 8 points correspond to lines of the far plane.
    std::vector< Vector4f > frustumLines() const;

    static PerspectiveCamera lerp( const PerspectiveCamera& c0, const PerspectiveCamera& c1, float t );
    static PerspectiveCamera cubicInterpolate( const PerspectiveCamera& c0, const PerspectiveCamera& c1, const PerspectiveCamera& c2, const PerspectiveCamera& c3, float t );

};
