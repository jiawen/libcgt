#pragma once

#include <vector>

#include <geometry/Plane3f.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>
#include <vecmath/Matrix4f.h>
#include <vecmath/Rect2f.h>

class BoundingBox3f;

class Camera
{
public:

	// Initializes to a camera at (0, 0, 5) looking at (0, 0, 0)
	// with zNear = 1, zFar = 100, an FoV of 50 degrees and an aspect ratio	of 1:1
	Camera( const Vector3f& eye = Vector3f( 0, 0, 5 ),
		const Vector3f& center = Vector3f( 0, 0, 0 ),
		const Vector3f& up = Vector3f( 0, 1, 0 ),
		float left = -0.46630767f, float right = 0.46630767f,
		float bottom = -0.46630767f, float top = 0.46630767f,
		float zNear = 1.0f, float zFar = 100.0f, bool zFarIsInfinite = false,
		bool isDirectX = true );

    void setDirectX( bool directX );

	void getFrustum( float* pfLeft, float* pfRight,
		float* pfBottom, float* pfTop,
		float* pfZNear, float* pfZFar,
		bool* pbZFarIsInfinite = nullptr ) const;

	// return the 8 corners of the camera frustum
	// 0-3 are the near plane, in ccw order
	// 4-7 are the far plane, in ccw order
	std::vector< Vector3f > frustumCorners() const;

	// return the 6 planes of the camera frustum
	// in the order: left, bottom, right, top, near, far
	// all the planes have normals pointing outward	
	std::vector< Plane3f > frustumPlanes() const;

	bool isZFarInfinite();

	void setFrustum( float left, float right,
		float bottom, float top,
		float zNear, float zFar,
		bool zFarIsInfinite = false );

	void getLookAt( Vector3f* pEye, Vector3f* pCenter, Vector3f* pUp ) const;

	// up should be of unit length
	void setLookAt( const Vector3f& eye,
		const Vector3f& center,
		const Vector3f& up );

    Vector3f eye() const { return m_eye; }
	void setEye( const Vector3f& eye );

	Vector3f center() const { return m_center; }
	void setCenter( const Vector3f& center );

	// return the "up" unit vector
	Vector3f up() const;
	void setUp( const Vector3f& up );

	// return the "forward" unit vector
	Vector3f forward() const;

	// return the "right" unit vector (forward cross up)
	Vector3f right() const;	
	
	float zNear() const;
	void setZNear( float zNear );

	float zFar() const;
	void setZFar( float zFar );

	virtual Matrix4f projectionMatrix() const = 0;

	// returns the projection matrix P such that
	// the plane at a distance focusZ in front of the center of the lens
	// is kept constant while the eye has been moved
	// by (eyeX, eyeY) *in the plane of the lens*
	// i.e. eyeX is in the direction of right()
	// and eyeY is in the direction of up()
	Matrix4f jitteredProjectionMatrix( float eyeX, float eyeY, float focusZ ) const;

	Matrix4f viewMatrix() const;

	// returns the view matrix V such that
	// the eye has been moved by (fEyeX, fEyeY)
	// *in the plane of the lens*
	// i.e. eyeX is in the direction of getRight()
	// and eyeY is in the direction of getUp()
	Matrix4f jitteredViewMatrix( float eyeX, float eyeY ) const;

	// equivalent to viewMatrix() * projectionMatrix()
	Matrix4f viewProjectionMatrix() const;
	
	// equivalent to jitteredProjectionMatrix() * jitteredViewMatrix()
	Matrix4f jitteredViewProjectionMatrix( float eyeX, float eyeY, float focusZ ) const;

	Matrix4f inverseProjectionMatrix() const;
	Matrix4f inverseViewMatrix() const;
	Matrix4f inverseViewProjectionMatrix() const;

	// Given a point in the world and a screen of size screenSize
	// returns (x,y,z,w) where:
	// (x,y) is the 2D pixel coordinate on the screen (outside [0,width),[0,height) is clipped)
	// z is the nonlinear screen-space z,
	//   where [zNear,zFar] is mapped to [0,1] (outside this range is clipped)
	// w is z-coordinate (not radial distance) in eye space (farther things are positive)
	// This is what comes out as the float4 SV_POSITION in an HLSL pixel shader
	Vector4f projectToScreen( const Vector4f& world, const Vector2i& screenSize );

	// TODO: write unprojectToWorld(), that takes in the useless zScreen value
	// just for completeness
	// pixelToEye and pixelToWorld are much more useful.
	// In OpenGL: zNDC = (f+n)/(f-n) + 2fn/(f-n) * (1/zEye), zEye = -depth

	// given a 2D pixel (x,y) on a screen of size screenSize
	// returns a 3D ray direction
	// (call eye() to get the ray origin)
	Vector3f pixelToDirection( const Vector2f& xy, const Vector2i& screenSize );

	// xy and viewport are in pixel coordinates
	Vector3f pixelToDirection( const Vector2f& xy, const Rect2f& viewport );	

	// TODO: support viewports
	// given a pixel (x,y) in screen space (in [0,screenSize.x), [0,screenSize.y))
	// returns its NDC in [-1,1]^2
	Vector2f pixelToNDC( const Vector2f& xy, const Vector2i& screenSize );	

	// given a pixel (x,y) in screen space (in [0,screenSize.x), [0,screenSize.y))
	// and an actual depth value (\in [0, +inf)), not distance along ray,
	// returns its eye space coordinates (right handed, output z will be negative), output.w = 1
	virtual Vector4f pixelToEye( const Vector2f& xy, float depth, const Vector2i& screenSize );

	// given a pixel (x,y) in screen space (in [0,screenSize.x), [0,screenSize.y))
	// and an actual depth value (\in [0, +inf)), not distance along ray,
	// returns its world space coordinates, output.w = 1
	Vector4f pixelToWorld( const Vector2f& xy, float depth, const Vector2i& screenSize );		

protected:

	float m_left;
	float m_right;
	
	float m_top;
	float m_bottom;

	float m_zNear;
	float m_zFar;
	bool m_zFarIsInfinite;

	Vector3f m_eye;
	Vector3f m_center;
	Vector3f m_up;

	// if the projection matrix is for DirectX or OpenGL
	// the view matrix is always right handed
	// the only difference in the projection matrix is whether
	// [zNear, zFar] gets mapped to [0,1] (DirectX) or [-1,1] (OpenGL) in NDC	
    bool m_directX;
};
