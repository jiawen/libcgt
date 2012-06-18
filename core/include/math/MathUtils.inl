#pragma once

// static
template< typename T >
T MathUtils::lerp( const T& x, const T& y, float t )
{
	return( x + t * ( y - x ) );
}

// static
inline float MathUtils::oo_0( float x )
{
	return x != 0 ? 1.0f / x : 0.0f;
}

// static
inline double MathUtils::oo_0( double x )
{
	return x != 0 ? 1.0 / x : 0.0;
}