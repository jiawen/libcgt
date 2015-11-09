#pragma once

#include <common/BasicTypes.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

// TODO: use std RNG
#include "MersenneTwister.h"

// simple wrapper around the Mersenne Twister
// hides a bunch of options and renames functions to be nicer
class Random
{
public:

    Random(); // from /dev/random or from clock()
    Random( int seed ); // seed from integer

    // [0, 2^32 - 1]
    uint32_t nextInt();

    // [0,n) for n < 2^32
    int nextIntExclusive( int n );

    // [0,n] for n < 2^32
    int nextIntInclusive( int n );

    // [0,1]
    double nextDouble();
    float nextFloat();

    // [0,1]^d
    Vector2f nextVector2f();
    Vector3f nextVector3f();
    Vector4f nextVector4f();

    // [lo, lo+count)
    int nextIntRange( int lo, int count );

    // [lo,hi]
    double nextDoubleRange( double lo, double hi );
    float nextFloatRange( float lo, float hi );

    // [lo,hi]^d
    Vector2f nextVector2fRange( const Vector2f& lo, const Vector2f& hi );
    Vector3f nextVector3fRange( const Vector3f& lo, const Vector3f& hi );
    Vector4f nextVector4fRange( const Vector4f& lo, const Vector4f& hi );

private:

    MTRand m_mtRand;

};
