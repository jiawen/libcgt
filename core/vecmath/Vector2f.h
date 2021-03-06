#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

class Vector2d;
class Vector2i;
class Vector3f;

class Vector2f
{
public:

    // TODO: conversion operators for double <--> float on Vector3f and Vector4f

    // Default constructor initializes to all 0s.
    Vector2f() = default;
    explicit Vector2f( float f ); // (f,f)
    Vector2f( float _x, float _y );

    // copy constructors
    Vector2f( const Vector2f& v ) = default;
    Vector2f& operator = ( const Vector2f& v ) = default;

    // Cast.
    explicit Vector2f( const Vector2d& rv );
    explicit Vector2f( const Vector2i& rv );

    // returns the ith element
    const float& operator [] ( int i ) const;
    float& operator [] ( int i );

    Vector2f xy() const;
    Vector2f yx() const;
    Vector2f xx() const;
    Vector2f yy() const;

    // returns ( -y, x )
    Vector2f normal() const;

    float norm() const;
    float normSquared() const;

    void normalize();
    Vector2f normalized() const;

    // ---- Utility ----
    operator const float* () const;
    operator float* ();
    std::string toString() const;

    static float dot( const Vector2f& v0, const Vector2f& v1 );

    // returns (0,0, x0 * y1 - x1 * y0 )
    static Vector3f cross( const Vector2f& v0, const Vector2f& v1 );

    Vector2f& operator += ( const Vector2f& v );
    Vector2f& operator -= ( const Vector2f& v );
    Vector2f& operator *= ( float f );
    Vector2f& operator /= ( float f );

    float x = 0.f;
    float y = 0.f;
};

Vector2f operator + ( const Vector2f& v0, const Vector2f& v1 );
Vector2f operator + ( const Vector2f& v, float f );
Vector2f operator + ( float f, const Vector2f& v );

Vector2f operator - ( const Vector2f& v0, const Vector2f& v1 );
Vector2f operator - ( const Vector2f& v, float f );
Vector2f operator - ( float f, const Vector2f& v );
// negate
Vector2f operator - ( const Vector2f& v );

Vector2f operator * ( float f, const Vector2f& v );
Vector2f operator * ( const Vector2f& v, float f );

// component-wise multiplication
Vector2f operator * ( const Vector2f& v0, const Vector2f& v1 );

// component-wise division
Vector2f operator / ( const Vector2f& v, float f );
Vector2f operator / ( const Vector2f& v0, const Vector2f& v1 );

// reciprocal of each component
Vector2f operator / ( float f, const Vector2f& v );

bool operator == ( const Vector2f& v0, const Vector2f& v1 );
bool operator != ( const Vector2f& v0, const Vector2f& v1 );

#include "libcgt/core/vecmath/Vector2d.h"
#include "libcgt/core/vecmath/Vector2i.h"
#include "libcgt/core/vecmath/Vector2f.inl"
