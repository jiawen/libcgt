#pragma once

#include <algorithm>

// static
template< typename T >
T MathUtils::lerp( const T& x, const T& y, float t )
{
    return( x + t * ( y - x ) );
}

// static
template< typename T >
T MathUtils::clampToRange( const T& x, const T& lo, const T& hi )
{
    return std::max( lo, std::min( x, hi ) );
}

// static
template< typename T >
T MathUtils::cubicInterpolate( const T& p0, const T& p1, const T& p2, const T& p3, float t )
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