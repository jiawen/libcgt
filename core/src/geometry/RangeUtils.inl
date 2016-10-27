namespace libcgt { namespace core { namespace geometry {

inline float rescale( float x, const Range1f& src, const Range1f& dst )
{
    assert( src.isStandard() );
    assert( !src.isEmpty() );
    assert( dst.isStandard() );
    assert( !dst.isEmpty() );

    return libcgt::core::math::lerp( dst,
        libcgt::core::math::fraction( x, src ) );
}

inline int rescale( float x, const Range1f& src, const Range1i& dst )
{
    assert( src.isStandard() );
    assert( !src.isEmpty() );
    assert( dst.isStandard() );
    assert( !dst.isEmpty() );

    return libcgt::core::math::roundToInt(
        libcgt::core::math::lerp( dst,
            libcgt::core::math::fraction( x, src ) ) );
}

inline float rescale( int x, const Range1i& src, const Range1f& dst )
{
    assert( src.isStandard() );
    assert( !src.isEmpty() );
    assert( dst.isStandard() );
    assert( !dst.isEmpty() );

    return libcgt::core::math::lerp( dst,
        libcgt::core::math::fraction( x, src ) );
}

inline int rescale( int x, const Range1i& src, const Range1i& dst )
{
    assert( src.isStandard() );
    assert( !src.isEmpty() );
    assert( dst.isStandard() );
    assert( !dst.isEmpty() );

    return libcgt::core::math::roundToInt(
        libcgt::core::math::lerp( dst,
            libcgt::core::math::fraction( x, src ) ) );
}

} } } // geometry, core, libcgt
