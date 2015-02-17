#include "math/Random.h"

Random::Random()
{

}

Random::Random( int seed ) :

	m_mtRand( seed )

{

}

double Random::nextDouble()
{
	return m_mtRand.rand();
}

float Random::nextFloat()
{
	return static_cast< float >( nextDouble() );
}

Vector2f Random::nextVector2f()
{
    return { nextFloat(), nextFloat() };
}

Vector3f Random::nextVector3f()
{
	return Vector3f( nextFloat(), nextFloat(), nextFloat() );
}

Vector4f Random::nextVector4f()
{
	return Vector4f( nextFloat(), nextFloat(), nextFloat(), nextFloat() );
}

int Random::nextIntRange( int lo, int count )
{
	return lo + nextIntExclusive( count );
}

double Random::nextDoubleRange( double lo, double hi )
{
	double range = hi - lo;
	return( lo + range * nextDouble() );
}

float Random::nextFloatRange( float lo, float hi )
{
	float range = hi - lo;
	return( lo + range * nextFloat() );
}

Vector2f Random::nextVector2fRange( const Vector2f& lo, const Vector2f& hi )
{
    return{ nextFloatRange( lo.x, hi.x ), nextFloatRange( lo.y, hi.y ) };
}

Vector3f Random::nextVector3fRange( const Vector3f& lo, const Vector3f& hi )
{
	return Vector3f( nextFloatRange( lo.x, hi.x ), nextFloatRange( lo.y, hi.y ),
		nextFloatRange( lo.z, hi.z ) );
}

Vector4f Random::nextVector4fRange( const Vector4f& lo, const Vector4f& hi )
{
	return Vector4f( nextFloatRange( lo.x, hi.x ), nextFloatRange( lo.y, hi.y ),
		nextFloatRange( lo.z, hi.z ), nextFloatRange( lo.w, hi.w ) );
}

uint32_t Random::nextInt()
{
	return m_mtRand.randInt();
}

int Random::nextIntExclusive( int n )
{
	return nextIntInclusive( n - 1 );
}

int Random::nextIntInclusive( int n )
{
	return m_mtRand.randInt( n );
}
