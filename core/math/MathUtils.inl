#pragma once

#include <algorithm>
#include <cassert>

#include "libcgt/core/vecmath/Range1f.h"
#include "libcgt/core/vecmath/Range1i.h"

namespace libcgt { namespace core { namespace math {

inline int clampToRangeExclusive( int x, int lo, int hi )
{
    assert( lo < hi );
    return std::max( lo, std::min( x, hi - 1 ) );
}

inline float clampToRangeInclusive( float x, float lo, float hi )
{
    assert( lo <= hi );
    return std::max( lo, std::min( x, hi ) );
}

inline double clampToRangeInclusive( double x, double lo, double hi )
{
    assert( lo <= hi );
    return std::max( lo, std::min( x, hi ) );
}

inline int clamp( int x, const Range1i& range )
{
    assert( range.isStandard() );
    return std::max( range.left(), std::min( x, range.right() - 1 ) );
}

inline float clamp( float x, const Range1f& range )
{
    assert( range.isStandard() );
    return std::max( range.left(), std::min( x, range.right() ) );
}

template< typename T >
inline T lerp( T x, T y, float t )
{
    return( x + t * ( y - x ) );
}

template< typename T >
inline T lerp( T x, T y, double t )
{
    return( x + t * ( y - x ) );
}

template<>
inline uint8x3 lerp( uint8x3 x, uint8x3 y, float t )
{
    // TODO: assert t is in [0, 1].
    // TODO: use toUInt8()
    int32_t t2 = static_cast< int32_t >( 255.0f * t );
    Vector3i x2( x.x, x.y, x.z );
    Vector3i y2( y.x, y.y, y.z );
    Vector3i d = y2 - x2;
    Vector3i r = t2 * d;
    return uint8x3
    {
        static_cast< uint8_t >( x.x + ( r.x >> 8 ) ),
        static_cast< uint8_t >( x.y + ( r.y >> 8 ) ),
        static_cast< uint8_t >( x.z + ( r.z >> 8 ) )
    };
}

inline float lerp( const Range1f& range, float t )
{
    assert( range.isStandard() );
    return( range.origin + t * range.size );
}

inline float lerp( const Range1i& range, float t )
{
    assert( range.isStandard() );
    return( range.origin + t * range.size );
}

template< typename T >
inline T cubicInterpolate( const T& p0, const T& p1, const T& p2, const T& p3,
    float t )
{
    // geometric construction:
    //            t
    //   (t+1)/2     t/2
    // t+1        t         t-1

    // bottom level
    T p0p1 = lerp( p0, p1, t + 1 );
    T p1p2 = lerp( p1, p2, t );
    T p2p3 = lerp( p2, p3, t - 1 );

    // middle level
    T p0p1_p1p2 = lerp( p0p1, p1p2, 0.5f * ( t + 1 ) );
    T p1p2_p2p3 = lerp( p1p2, p2p3, 0.5f * t );

    // top level
    return lerp( p0p1_p1p2, p1p2_p2p3, t );
}

inline float fraction( float x, const Range1f& range )
{
    assert( range.isStandard() );
    assert( !range.isEmpty() );
    return ( x - range.origin ) / ( range.size );
}

inline float fraction( int x, const Range1i& range )
{
    assert( range.isStandard() );
    assert( !range.isEmpty() );
    return static_cast< float >( x - range.origin ) / ( range.size );
}


inline float oo_0( float x )
{
    return x != 0 ? 1.0f / x : 0.0f;
}

inline double oo_0( double x )
{
    return x != 0 ? 1.0 / x : 0.0;
}

} } } // math, core, libcgt
