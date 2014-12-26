#pragma once

#include <vector>

#include "geometry/Plane3f.h"
#include "vecmath/Vector2f.h"
#include "vecmath/Vector2f.h"
#include "vecmath/Vector3f.h"
#include "vecmath/Vector4f.h"
#include "vecmath/Matrix4f.h"
#include "vecmath/Rect2f.h"

class Camera
{
public:

    void setDirectX( bool directX );

	void getFrustum( float& left, float& right,
		float& bottom, float& top,
		float& zNear, float& zFar ) const;

	// return the 8 corners of the camera frustum
	// 0-3 are the near plane, in ccw order
	// 4-7 are the far plane, in ccw order
	std::vector< Vector3f > frustumCorners() const;

	// return the 6 planes of the camera frustum
	// in the order: left, bottom, right, top, near, far
	// all the planes have normals pointing outward	
	std::vector< Plane3f > frustumPlanes() const;

	bool isZFarInfinite() const;

    // Sets the camera frustum, OpenGL-style (glFrustum()).
    // zFar can be std::numeric_limits< float >::infinity()
    // to get a proper infinite view frustum.
	void setFrustum( float left, float right,
		float bottom, float top,
		float zNear, float zFar );

	// same as below, but uses existing zNear and zFar
	void setFrustumFromIntrinsics( const Vector2f& focalLengthPixels, const Vector2f& principalPointPixels,
		const Vector2f& imageSize );

	// Sets the perspective projection given computer-vision style intrinsics:
	// focal length and image size in pixels
	// Does not allow for radial distortion
	void setFrustumFromIntrinsics( const Vector2f& focalLengthPixels, const Vector2f& principalPointPixels,
		const Vector2f& imageSize,
		float zNear, float zFar );

    // Returns the three OpenGL style lookat parameters:
    // the eye point, the center point (eye + distance * forward), and the up vector.
	void getLookAt( Vector3f& eye, Vector3f& center, Vector3f& up ) const;

    // Sets the camera position with an OpenGL-style gluLookAt().
    // eye is a point, center is a point towards which the camera is pointing.
    // (center - eye) is not necessary a unit vector.
    // forward is (center - eye).normalized()
    // the z axis is -forward().
    // Up should be a unit vector that's orthogonal to (center - eye).
	void setLookAt( const Vector3f& eye,
		const Vector3f& center,
		const Vector3f& up );

    // Sets the view matrix given an "inverse view matrix"
    // ivm: the mapping from eye -> world.
	void setLookAtFromInverseViewMatrix( const Matrix4f& ivm );

    Vector3f eye() const;
	void setEye( const Vector3f& eye );

	Vector3f center() const;
	void setCenter( const Vector3f& center );

	// return the "up" unit vector
	Vector3f up() const;
	void setUp( const Vector3f& up );

	// Returns the "forward" unit vector, equivalent to
    // ( center() - eye() ).normalized().
	Vector3f forward() const;

	// Returns the "right" unit vector, equivalent to
    // cross( forward(), up() ).
	Vector3f right() const;	
	
	float zNear() const;
	void setZNear( float zNear );

	float zFar() const;
	void setZFar( float zFar );

	// TODO: virtual QString toString() const;

    // Returns the OpenGL / Direct3D style projection matrix,
    // mapping eye space to clip space.
	virtual Matrix4f projectionMatrix() const = 0;
	
    // Returns the OpenGL / Direct3D style view matrix,
    // mapping world space to eye space.
	Matrix4f viewMatrix() const;

	// returns the view matrix V such that
	// the eye has been moved by (fEyeX, fEyeY)
	// *in the plane of the lens*
	// i.e. eyeX is in the direction of getRight()
	// and eyeY is in the direction of getUp()
	Matrix4f jitteredViewMatrix( float eyeX, float eyeY ) const;

	// equivalent to viewMatrix() * projectionMatrix()
	Matrix4f viewProjectionMatrix() const;	

	Matrix4f inverseProjectionMatrix() const;
	Matrix4f inverseViewMatrix() const;
	Matrix4f inverseViewProjectionMatrix() const;

    // Returns computer vision camera extrinsics:
    // Directly mapping GL-convention world space points
    // to CV-convention eye space points.
    Matrix4f extrinsicsCG2CV() const;

    // Returns computer vision camera extrinsics:
    // a matrix mapping points in CV-convention world space
    // to CV-convention eye space.
    //
    // Recall that CV-convention coordinates are right handed
    // and have x right, y down, z into the screen. To use
    // this matrix correctly, input points in world space should
    // also have this convention.
    //
    // If your world coordinates are GL-convention, with
    // x right, y up, and z out of the screen, convert them
    // to CV-convention world coordinates by first multiplying
    // by Matrix4f::ROTATE_X_180. Or, directly go from GL world
    // coordinates to CV eye coordinates with extrinsicsCG2CV().
    //
    // The bottom row is [0 0 0 1] and can be ignored
    // if the CV convention is a 3x4 matrix.
    //
    // If the CV library uses a Rodrigues rotation vector,
    // use Quat4f::fromRotationMatrix(extrinsics().getSubmatrix3x3()).toAxisAngle().
    // And multiply the unit vector by the angle in radians.
    // TODO: directly return rotation vector.
    //
    // Since OpenCV's coordinate convention is right handed,
    // but with x right, y down, z into the screen,
    // this matrix is equivalent *conjugated* by rotX(pi):
    // to rotX(pi)^-1 * viewMatrix() * rotX(pi).
    // But rotX(pi)^-1 = rotx(pi).
    // This of the operation as:
    // Given a point in the CV world:
    // - apply rotX(pi) to bring it into the GL world
    // - apply the view matrix in the GL world
    // - apply rotX(pi) again to bring it back into the CV world
    Matrix4f extrinsicsCV() const;

    // Returns the CV-convention camera intrinsics:
    // A matrix mapping homogenized CV-convention eye-space points
    // ( (x_eye_cv / z_eye_cv), (y_eye_cv / z_eye_cv)) to CV-convention
    // pixels in screen space.
    //
    // CV-convention eye space has x right, y down, z into the screen,
    // as does CV-convension screen space pixels.
    Matrix3f intrinsicsCV( const Vector2f& screenSize ) const;

	Vector4f worldToEye( const Vector4f& world ) const;
	Vector4f eyeToScreen( const Vector4f& eye, const Vector2f& screenSize ) const;

	// Given a point in the world and a screen of size screenSize
	// returns (x,y,z,w) where:
	// (x,y) is the 2D pixel coordinate on the screen (outside [0,width),[0,height) is clipped)
	// z is the nonlinear screen-space z,
	//   where [zNear,zFar] is mapped to [0,1] (outside this range is clipped)
	// w is z-coordinate (not radial distance) in eye space (farther things are positive)
	// This is what comes out as the float4 SV_POSITION in an HLSL pixel shader
	Vector4f worldToScreen( const Vector4f& world, const Vector2f& screenSize ) const;

	// TODO: write unprojectToWorld(), that takes in the useless zScreen value
	// just for completeness
	// pixelToEye and pixelToWorld are much more useful.
	// In OpenGL: zNDC = (f+n)/(f-n) + 2fn/(f-n) * (1/zEye), zEye = -depth

	// given a 2D pixel (x,y) on a screen of size screenSize
	// returns a 3D ray direction
	// (call eye() to get the ray origin)
	// (integer versions are at the center of pixel)
	Vector3f pixelToDirection( const Vector2i& xy, const Vector2f& screenSize ) const;
	Vector3f pixelToDirection( const Vector2f& xy, const Vector2f& screenSize ) const;

	// xy and viewport are in pixel coordinates
	Vector3f pixelToDirection( const Vector2f& xy, const Rect2f& viewport ) const;

	// TODO: support viewports
	// given a pixel (x,y) in screen space (in [0,screenSize.x), [0,screenSize.y))
	// returns its NDC in [-1,1]^2
	Vector2f pixelToNDC( const Vector2f& xy, const Vector2f& screenSize ) const;

	// given a pixel (x,y) in screen space (in [0,screenSize.x), [0,screenSize.y))
	// and an actual depth value (\in [0, +inf)), not distance along ray,
	// returns its eye space coordinates (right handed, output z will be negative), output.w = 1
	// (integer versions are at the center of pixel)
	virtual Vector4f pixelToEye( const Vector2i& xy, float depth, const Vector2f& screenSize ) const;
	virtual Vector4f pixelToEye( const Vector2f& xy, float depth, const Vector2f& screenSize ) const;

	// given a pixel (x,y) in screen space (in [0,screenSize.x), [0,screenSize.y))
	// and an actual depth value (\in [0, +inf)), not distance along ray,
	// returns its world space coordinates, output.w = 1
	// (integer versions are at the center of pixel)
	Vector4f pixelToWorld( const Vector2i& xy, float depth, const Vector2f& screenSize ) const;
	Vector4f pixelToWorld( const Vector2f& xy, float depth, const Vector2f& screenSize ) const;

protected:

	float m_left;
	float m_right;
	
	float m_top;
	float m_bottom;

	float m_zNear;
	float m_zFar;

	Vector3f m_eye;
	Vector3f m_center;
	Vector3f m_up;

	// if the projection matrix is for DirectX or OpenGL
	// the view matrix is always right handed
	// the only difference in the projection matrix is whether
	// [zNear, zFar] gets mapped to [0,1] (DirectX) or [-1,1] (OpenGL) in NDC	
    bool m_directX;
};
