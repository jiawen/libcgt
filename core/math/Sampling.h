#pragma once

class Random;
class SamplingPatternND;

#include "libcgt/core/vecmath/Vector2f.h"
#include "libcgt/core/vecmath/Vector3f.h"

// TODO(jiawen): convert to pure functions
class Sampling
{
public:

    // TODO(VS2015): return an Array1DReadView< Vector< float, dim > > or
    // something.
    // populates pPattern with a latin hypercube sampling pattern
    static void latinHypercubeSampling( Random& random,
        SamplingPatternND* pPattern );

    // Given uniform random numbers u0, u1 in [0,1]
    // Returns a point uniformly sampled
    // over the area of the unit disc (center 0, radius 1)
    static Vector2f areaSampleDisc( float u0, float u1 );

    // Given uniform random numbers u0, u1 in [0,1]
    // Returns a point concentrically sampled
    // over the area of the unit disc (center 0, radius 1)
    //
    // Compared to areaSampleDisc, the distribution is
    // less distorted and better preserves distances between points
    static Vector2f concentricSampleDisc( float u1, float u2 );

    // Given uniform random number u0 in [0,1]
    // Returns a point uniformly sampled
    // over the perimeter of the unit circle (center 0, radius 1)
    static Vector2f perimeterSampleCircle( float u0 );

    // Given uniform random numbers u0, u1 in [0,1]
    // Returns a point uniformly sampled
    // over the surface area of the unit sphere (center 0, radius 1)
    static Vector3f areaSampleSphere( float u0, float u1 );

    // Given uniform random numbers u0, u1 in [0,1]
    // returns the *barycentric coordinates* of a random point in an arbitrary triangle
    static Vector3f areaSampleTriangle( float u0, float u1 );

};
