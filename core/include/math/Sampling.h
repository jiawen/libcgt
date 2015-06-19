#pragma once

class Random;
class SamplingPatternND;

#include "vecmath/Vector2f.h"
#include "vecmath/Vector3f.h"

class Sampling
{
public:

	// TODO: fix this, and sampling pattern
	// the interface sucks!

	// populates pPattern with a latin hypercube sampling pattern
	static void latinHypercubeSampling( Random& random, SamplingPatternND* pPattern );

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
