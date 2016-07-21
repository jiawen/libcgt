#pragma once

#include <algorithm>
#include <cassert>
#include <vecmath/Range1f.h>
#include <vecmath/Range1i.h>

namespace libcgt { namespace core { namespace math {

int clampToRangeExclusive( int x, int lo, int hi )
{
    assert( lo < hi );
    return std::max( lo, std::min( x, hi - 1 ) );
}

float clampToRangeInclusive( float x, float lo, float hi )
{
    assert( lo <= hi );
    return std::max( lo, std::min( x, hi ) );
}

double clampToRangeInclusive( double x, double lo, double hi )
{
    assert( lo <= hi );
    return std::max( lo, std::min( x, hi ) );
}

int clamp( int x, const Range1i& range )
{
    assert( range.isStandard() );
    return std::max( range.left(), std::min( x, range.right() - 1 ) );
}

float clamp( float x, const Range1f& range )
{
    assert( range.isStandard() );
    return std::max( range.left(), std::min( x, range.right() ) );
}

template< typename T >
T lerp( const T& x, const T& y, float t )
{
    return( x + t * ( y - x ) );
}

template< typename T >
T lerp( const T& x, const T& y, double t )
{
    return( x + t * ( y - x ) );
}

float lerp( const Range1f& range, float t )
{
    assert( range.isStandard() );
    return( range.origin + t * range.size );
}

float lerp( const Range1i& range, float t )
{
    assert( range.isStandard() );
    return( range.origin + t * range.size );
}

template< typename T >
T cubicInterpolate( const T& p0, const T& p1, const T& p2, const T& p3,
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

float fraction( float x, const Range1f& range )
{
    assert( range.isStandard() );
    assert( !range.isEmpty() );
    return ( x - range.origin ) / ( range.size );
}

float fraction( int x, const Range1i& range )
{
    assert( range.isStandard() );
    assert( !range.isEmpty() );
    return static_cast< float >( x - range.origin ) / ( range.size );
}

float rescale( float x, const Range1f& src, const Range1f& dst )
{
    assert( src.isStandard() );
    assert( !src.isEmpty() );
    assert( dst.isStandard() );
    assert( !dst.isEmpty() );

    return lerp( dst, fraction( x, src ) );
}

int rescale( float x, const Range1f& src, const Range1i& dst )
{
    assert( src.isStandard() );
    assert( !src.isEmpty() );
    assert( dst.isStandard() );
    assert( !dst.isEmpty() );

    return roundToInt( lerp( dst, fraction( x, src ) ) );
}

float rescale( int x, const Range1i& src, const Range1f& dst )
{
    assert( src.isStandard() );
    assert( !src.isEmpty() );
    assert( dst.isStandard() );
    assert( !dst.isEmpty() );

    return lerp( dst, fraction( x, src ) );
}

int rescale( int x, const Range1i& src, const Range1i& dst )
{
    assert( src.isStandard() );
    assert( !src.isEmpty() );
    assert( dst.isStandard() );
    assert( !dst.isEmpty() );

    return roundToInt( lerp( dst, fraction( x, src ) ) );
}

float oo_0( float x )
{
    return x != 0 ? 1.0f / x : 0.0f;
}

double oo_0( double x )
{
    return x != 0 ? 1.0 / x : 0.0;
}

} } } // math, core, libcgt
