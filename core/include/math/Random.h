#ifndef RANDOM_H
#define RANDOM_H

#include <common/BasicTypes.h>

#include "MersenneTwister.h"

// simple wrapper around the Mersenne Twister
// hides a bunch of options and renames functions to be nicer
class Random
{
public:

	Random(); // from /dev/random or from clock()
	Random( int seed ); // seed from integer

	// [0,1]
	double nextDouble() const;

	// [0,1]
	float nextFloat() const;

	// [0, 2^32 - 1]
	uint nextInt() const;

	double nextDoubleRange( double lo, double hi ) const;

	float nextFloatRange( float lo, float hi ) const;

	// [0,n] for n < 2^32
	int nextIntInclusive( int n ) const;

	// [0,n) for n < 2^32
	int nextIntExclusive( int n ) const;

private:

	mutable MTRand m_mtRand;

};

#endif // RANDOM_H
