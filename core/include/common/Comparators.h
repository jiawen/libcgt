#pragma once

#include <utility>

class Vector2i;
class Vector3i;
class Vector4i;

// TODO: sort two parallel vectors
class Comparators
{
public:
	
	// define the Vector <N>i Comparator types (function pointer type)
	typedef bool (*Vector2iComparator)( const Vector2i&, const Vector2i& );
	typedef bool (*Vector3iComparator)( const Vector3i&, const Vector3i& );
	typedef bool (*Vector4iComparator)( const Vector4i&, const Vector4i& );

	// compares only the first element of a pair
	template< typename T0, typename T1 >
	static bool pairFirstElementLess( const std::pair< T0, T1 >& a, const std::pair< T0, T1 >& b );

	// compares only the second element of a pair
	template< typename T0, typename T1 >
	static bool pairSecondElementLess( const std::pair< T0, T1 >& a, const std::pair< T0, T1 >& b );

	// Returns true if x.second < y.second.
	// Useful for sorting array indices based on distance.
	static bool indexAndDistanceLess( const std::pair< int, float >& a, const std::pair< int, float >& b );	

    // Returns true if x.second > y.second.
	// Useful for sorting array indices based on distance.
    static bool indexAndDistanceGreater( const std::pair< int, float >& a, const std::pair< int, float >& b );	

	static bool vector2iLexigraphicLess( const Vector2i& a, const Vector2i& b );

	static bool vector3iLexigraphicLess( const Vector3i& a, const Vector3i& b );

	static bool vector4iLexigraphicLess( const Vector4i& a, const Vector4i& b );
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
