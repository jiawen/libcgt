#ifndef SAMPLING_H
#define SAMPLING_H

class Random;
class SamplingPatternND;

class Sampling
{
public:

	// TODO: fix this, and sampling pattern
	// the interface sucks!

	// populates pPattern with a latin hypercube sampling pattern
	static void latinHypercubeSampling( const Random& random, SamplingPatternND* pPattern );

	static void uniformSampleDisc( float u1, float u2,
		float* px, float* py );

	static void concentricSampleDisc( float u1, float u2,
		float* px, float* py );

};

#endif
