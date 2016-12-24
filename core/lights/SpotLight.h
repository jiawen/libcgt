#pragma once

#include <string>
#include <vector>

#include "libcgt/core/math/MathUtils.h"

#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Matrix3f.h"
#include "libcgt/core/vecmath/Matrix4f.h"

// TODO: consider just extending PerspectiveCamera.
class SpotLight
{
public:

    // TODO: variable aspect ratio
    // TODO: infinite frustum
    // TODO: cache matrices

    SpotLight( const Vector3f& position = Vector3f( 0, 5, 0 ),
        const Vector3f& center = Vector3f( 0, 0, 0 ),
        const Vector3f& up = Vector3f( 0, 0, 1 ),
        float fovYRadians = libcgt::core::math::degreesToRadians( 50.f ),
        float zNear = 1.0f, float fZFar = 100.0f, float aspect = 1.0f,
        const Vector3f& color = Vector3f( 1, 1, 1 ),
        const Vector3f& distanceFalloff = Vector3f( 0, 1, 0 ) );

    Vector3f color() const;
    void setColor( const Vector3f& color );

    Vector3f distanceFalloff() const;
    void setDistanceFalloff( const Vector3f& distanceFalloff );

    void setLookAt( const Vector3f& position,
        const Vector3f& center,
        const Vector3f& up );

    const Vector3f& position() const;
    void setPosition( const Vector3f& position );

    const Vector3f& center() const;
    void setCenter( const Vector3f& center );

    const Vector3f& up() const;
    void setUp( const Vector3f& up );

    // For a light that already has
    // a position and a center
    // compute the new up vector given a convenient right vector
    // equivalent to setUp( right cross lightDirection )
    void setUpWithRight( const Vector3f& right );

    float fovYRadians() const;
    void setFovYRadians( float fov );

    float aspectRatio() const;
    void setAspectRatio( float a );

    float zNear() const;
    void setZNear( float zNear );

    float zFar() const;
    void setZFar( float zFar );

    // (0,0,-1) in local coordinates
    Vector3f lightDirection() const;

    // (1,0,0) in local coordinates
    Vector3f right() const;

    // world --> clip
    Matrix4f lightProjectionMatrix() const;

    // clip --> world
    Matrix4f inverseLightProjectionMatrix() const;

    // world --> light
    Matrix4f lightMatrix() const;

    // light --> world
    Matrix4f inverseLightMatrix() const;

    // light --> clip
    Matrix4f projectionMatrix() const;

    // gives the vertices of a rectangle z units in front of the light position
    void rectangleAlignedAt( float z,
        Vector3f* bottomLeft, Vector3f* bottomRight,
        Vector3f* topRight, Vector3f* topLeft ) const;

    std::vector< Vector3f > getFrustumCorners() const;

    std::string toString() const;

private:

    Vector3f m_position;
    Vector3f m_center;
    Vector3f m_up;

    float m_fovYRadians;
    float m_zNear;
    float m_zFar;

    float m_aspect;

    Vector3f m_color;
    Vector3f m_distanceFalloff;
};
