#pragma once

#include <string>
#include <vector>

#include "cameras/GLFrustum.h"
#include "cameras/Intrinsics.h"

#include "geometry/Plane3f.h"
#include "vecmath/EuclideanTransform.h"
#include "vecmath/Vector2f.h"
#include "vecmath/Vector2f.h"
#include "vecmath/Vector3f.h"
#include "vecmath/Vector4f.h"
#include "vecmath/Matrix4f.h"
#include "vecmath/Rect2f.h"

class Camera
{
public:

    using EuclideanTransform = libcgt::core::vecmath::EuclideanTransform;
    using GLFrustum = libcgt::core::cameras::GLFrustum;
    using Intrinsics = libcgt::core::cameras::Intrinsics;

    // TODO(jiawen): remove this.
    bool isDirectX() const;

    // ---------------- Intrinsics ----------------

    const GLFrustum& frustum() const;

    // The image aspect ratio (width divided by height).
    float aspectRatio() const;

    // The (strictly positive) distance to the near plane.
    float zNear() const;

    // The distance to the far plane. May be infinite.
    float zFar() const;

    bool isZFarInfinite() const;

    // Returns the OpenGL / Direct3D style projection matrix,
    // mapping eye space to clip space.
    // TODO(jiawen): get it directly from the frustum, instead of virtual.
    virtual Matrix4f projectionMatrix() const = 0;

    Matrix4f inverseProjectionMatrix() const;

    // Get the 8 corners of the camera frustum.
    // 0-3 are the near plane, in ccw order.
    // 4-7 are the far plane, in ccw order.
    std::vector< Vector3f > frustumCorners() const;

    // Get the 6 planes of the camera frustum,
    // in the order: left, bottom, right, top, near, far
    // all the planes have normals pointing outward.
    std::vector< Plane3f > frustumPlanes() const;

    // ---------------- Extrinsics ----------------

    EuclideanTransform cameraFromWorld() const;
    void setCameraFromWorld( const EuclideanTransform& cfw );

    // Set the camera from world (view) matrix directly.
    // This function assumes that cfw is a Euclidean transformation.
    void setCameraFromWorldMatrix(const Matrix4f& cfw);

    EuclideanTransform worldFromCamera() const;
    void setWorldFromCamera( const EuclideanTransform& wfc );

    // Set the world from camera (inverse view) matrix directly
    // (this is not common).
    // This function assumes that wfc is a Euclidean transformation.
    void setWorldFromCameraMatrix( const Matrix4f& wfc );

    // Sets the camera pose with an OpenGL-style gluLookAt().
    // eye is a point, center is a point towards which the camera is pointing.
    // (center - eye) does not have to be a unit vector, but will be normalized
    // when stored.
    // The z axis will be -forward().
    // up should be a unit vector that's orthogonal to (center - eye).
    void setLookAt( const Vector3f& eye, const Vector3f& center,
        const Vector3f& up );

    // The camera origin, in world coordinates.
    Vector3f eye() const;

    // Set the camera origin, in world coordinates.
    void setEye( const Vector3f& eye );

    // Returns the "right" unit vector, the camera x-axis, in the world frame.
    Vector3f right() const;

    // Return the "up" unit vector, the camera y-axis, in the world frame.
    Vector3f up() const;

    // Return the "back" unit vector, the camera z-axis, in the world frame.
    Vector3f back() const;

    // Returns the "forward" unit vector, the camera negative z-axis,
    // in the world frame.
    Vector3f forward() const;

    // Get the OpenGL / Direct3D style view matrix,
    // mapping world space to eye space.
    // (Equivalent to cameraFromWorld, but as a Matrix4f).
    Matrix4f viewMatrix() const;

    // Get the inverse of the view matrix, mapping
    // eye space back to world space.
    // (Equivalent to worldFromCamera, but as a Matrix4f).
    Matrix4f inverseViewMatrix() const;

    // Returns the view matrix V such that
    // the eye has been moved by (fEyeX, fEyeY)
    // *in the plane of the lens*
    // i.e. eyeX is in the direction of getRight()
    // and eyeY is in the direction of getUp().
    Matrix4f jitteredViewMatrix( float eyeX, float eyeY ) const;

    // Equivalent to viewMatrix() * projectionMatrix().
    Matrix4f viewProjectionMatrix() const;

    Matrix4f inverseViewProjectionMatrix() const;

    // Returns the camera intrinsics, a 3x3 matrix mapping eye-space rays to
    // pixel-space points.
    //
    // These intrinsics retain OpenGL-convention right-handed coordinates,
    // with x-right, y-up, z-towards-viewer.
    Intrinsics intrinsics( const Vector2f& screenSize ) const;

    // ----- projection: world --> eye --> clip --> NDC --> screen -----

    // TODO: rename these to yFromX conventions.

    // Given a point in world coordinates, transforms it into eye coordinates
    // by computing viewMatrix() * world.
    Vector4f worldToEye( const Vector4f& world ) const;

    // Given a point in eye coordinates, transforms it into clip coordinates
    // by computing projectionMatrix() * eye.
    Vector4f eyeToClip( const Vector4f& eye ) const;

    // Given a point in clip coordinates, transforms it into normalized device
    // coordinates by computing:
    // ndc = clip.homogenized(); ndc.w = clip.w;
    // In OpenGL: ndc.xyz \in [-1,1]^3.
    // In DirectX: ndc.xy \in [-1,1]^2, ndc.z \in [0,1].
    // ndc.w is the orthogonal depth from the eye (not along ray).
    Vector4f clipToNDC( const Vector4f& clip ) const;

    // Rescales normalized device coordinates NDC to screen coordinates (pixels),
    // given the screen size.
    // (x,y) is the 2D pixel coordinate on the screen.
    //   (outside [0,width),[0,height) is clipped)
    // z is the nonlinear screen-space z, where [zNear,zFar] is mapped to [0,1]
    //   (outside this range is clipped).
    // output.w = ndc.w (the w coordinate is passed through).
    // You should pass clip.w as ndc.w, where clip = projectionMatrix() * eye as
    //   in eyeToClip().
    // TODO: support viewports and depth range (aka 3D viewport).
    Vector4f ndcToScreen( const Vector4f& ndc, const Vector2f& screenSize ) const;

    // Composition of eyeToClip(), clipToNDC(), and ndcToScreen().
    Vector4f eyeToScreen( const Vector4f& eye, const Vector2f& screenSize ) const;

    // The composition of worldToEye() followed by eyeToScreen().
    Vector4f worldToScreen( const Vector4f& world, const Vector2f& screenSize ) const;

    // ----- unprojection: screen --> eye --> world -----

    // TODO: PerspectiveCamera::pixelToEye() might be the same thing
    //   may be able to remove and get rid of virtual

    // given a pixel (x,y) in screen space (in [0,screenSize.x), [0,screenSize.y))
    // and an actual depth value (\in [0, +inf)), not distance along ray,
    // returns its eye space coordinates (right handed, output z will be negative), output.w = 1
    // (integer versions are at the center of pixel)
    virtual Vector4f screenToEye( const Vector2i& xy, float depth, const Vector2f& screenSize ) const;
    virtual Vector4f screenToEye( const Vector2f& xy, float depth, const Vector2f& screenSize ) const;

    // given a pixel (x,y) in screen space (in [0,screenSize.x), [0,screenSize.y))
    // and an actual depth value (\in [0, +inf)), not distance along ray,
    // returns its world space coordinates, output.w = 1
    // (integer versions are at the center of pixel)
    Vector4f screenToWorld( const Vector2i& xy, float depth, const Vector2f& screenSize ) const;
    Vector4f screenToWorld( const Vector2f& xy, float depth, const Vector2f& screenSize ) const;

    // ----- unprojection: screen --> a ray in world space for raytracing -----
    // TODO: write unprojectToWorld(), that takes in the useless zScreen value
    // just for completeness
    // pixelToEye and pixelToWorld are much more useful.
    // In OpenGL: zNDC = (f+n)/(f-n) + 2fn/(f-n) * (1/zEye), zEye = -depth

    // Converts a 2D pixel (x,y) on a screen of size "screenSize"
    // to a 3D ray direction.
    Vector3f screenToDirection( const Vector2f& xy, const Vector2f& screenSize ) const;

    // Converts a 2D pixel (x,y) on a screen of size "screenSize"
    // to a 3D ray direction.
    // The integer pixel coordinates (xy) places the point at the center of the pixel.
    Vector3f screenToDirection( const Vector2i& xy, const Vector2f& screenSize ) const;

    // xy and viewport are in pixel coordinates
    Vector3f screenToDirection( const Vector2f& xy, const Rect2f& viewport ) const;

    virtual std::string toString() const = 0;

    // TODO: support viewports
    // Given a pixel (x,y) in screen space (\in [0,screenSize.x), [0,screenSize.y))
    // returns its NDC \in [-1,1]^2.
    static Vector2f screenToNDC( const Vector2f& xy, const Vector2f& screenSize );

    // Copies the camera frustum (intrinsics), from --> to.
    static void copyFrustum( const Camera& from, Camera& to );

    // Copies the camera pose (extrinsics), from --> to.
    static void copyPose( const Camera& from, Camera& to );

protected:

    void setFrustum( const GLFrustum& frustum );

    // TODO(jiawen): remove this.
    void setDirectX( bool directX );

private:

    EuclideanTransform m_cameraFromWorld;
    GLFrustum m_frustum; // TODO(jiawen): rename class to Frustum?

    // if the projection matrix is for DirectX or OpenGL
    // the view matrix is always right handed
    // the only difference in the projection matrix is whether
    // [zNear, zFar] gets mapped to [0,1] (DirectX) or [-1,1] (OpenGL) in NDC
    bool m_directX;
};
