#pragma once

#include <utility>

class Vector2i;

class Comparators
{
public:
	
	template< typename T0, typename T1 >
	static bool pairFirstElementLess( const std::pair< T0, T1 >& a, const std::pair< T0, T1 >& b );

	template< typename T0, typename T1 >
	static bool pairSecondElementLess( const std::pair< T0, T1 >& a, const std::pair< T0, T1 >& b );

	// returns true if x.second < y.second
	// Useful for sorting array indices based on distance.
	static bool indexAndDistanceLess( const std::pair< int, float >& a, const std::pair< int, float >& b );	

	static bool vector2iLexigraphicLess( const Vector2i& a, const Vector2i& b );
};

// static
template< typename T0, typename T1 >
bool Comparators::pairFirstElementLess( const std::pair< T0, T1 >& a, const std::pair< T0, T1 >& b )
{
	return( a.first < b.first );
}

// static
template< typename T0, typename T1 >
bool Comparators::pairSecondElementLess( const std::pair< T0, T1 >& a, const std::pair< T0, T1 >& b )
{
	return( a.second < b.second );
}
