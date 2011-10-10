#include "math/Random.h"

Random::Random()
{

}

Random::Random( int seed ) :

	m_mtRand( seed )

{

}

double Random::nextDouble() const
{
	return m_mtRand.rand();
}

float Random::nextFloat() const
{
	return static_cast< float >( nextDouble() );
}
	
uint Random::nextInt() const
{
	return m_mtRand.randInt();
}

double Random::nextDoubleRange( double lo, double hi ) const
{
	double range = hi - lo;
	return( lo + range * nextDouble() );
}

float Random::nextFloatRange( float lo, float hi ) const
{
	float range = hi - lo;
	return( lo + range * nextFloat() );
}

int Random::nextIntInclusive( int n ) const
{
	return m_mtRand.randInt( n );
}

int Random::nextIntExclusive( int n ) const
{
	return nextIntInclusive( n - 1 );
}
