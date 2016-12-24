#pragma once

#include <utility>

#include "libcgt/core/vecmath/Vector2i.h"
#include "libcgt/core/vecmath/Vector3i.h"
#include "libcgt/core/vecmath/Vector4i.h"

// TODO(jiawen): sort two parallel vectors
namespace libcgt { namespace core {

// Define a function pointer type for a comparator on Vector<N>i.
typedef bool (*Vector2iComparator)( const Vector2i&, const Vector2i& );
typedef bool (*Vector3iComparator)( const Vector3i&, const Vector3i& );
typedef bool (*Vector4iComparator)( const Vector4i&, const Vector4i& );

// Compares only the first element of a pair.
template< typename T0, typename T1 >
bool pairFirstElementLess( const std::pair< T0, T1 >& a,
    const std::pair< T0, T1 >& b );

// Compares only the second element of a pair.
template< typename T0, typename T1 >
bool pairSecondElementLess( const std::pair< T0, T1 >& a,
    const std::pair< T0, T1 >& b );

// Returns true if x.second < y.second.
// Useful for sorting array indices based on distance.
bool indexAndDistanceLess( const std::pair< int, float >& a,
    const std::pair< int, float >& b );

// Returns true if x.second > y.second.
// Useful for sorting array indices based on distance.
bool indexAndDistanceGreater( const std::pair< int, float >& a,
    const std::pair< int, float >& b );

bool lexigraphicLess( const Vector2i& a, const Vector2i& b );

bool lexigraphicLess( const Vector3i& a, const Vector3i& b );

bool lexigraphicLess( const Vector4i& a, const Vector4i& b );

} } // core, libcgt

#include "Comparators.inl"
