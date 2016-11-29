namespace libcgt { namespace core {

template< typename T0, typename T1 >
inline bool pairFirstElementLess( const std::pair< T0, T1 >& a,
    const std::pair< T0, T1 >& b )
{
    return( a.first < b.first );
}

template< typename T0, typename T1 >
inline bool pairSecondElementLess( const std::pair< T0, T1 >& a,
    const std::pair< T0, T1 >& b )
{
    return( a.second < b.second );
}

inline bool indexAndDistanceLess( const std::pair< int, float >& a, const std::pair< int, float >& b )
{
    return a.second < b.second;
}

inline bool indexAndDistanceGreater( const std::pair< int, float >& a,
    const std::pair< int, float >& b )
{
    return a.second > b.second;
}

inline bool lexigraphicLess( const Vector2i& a, const Vector2i& b )
{
    if( a.x < b.x )
    {
        return true;
    }
    else if( a.x > b.x )
    {
        return false;
    }
    else
    {
        return a.y < b.y;
    }
}

inline bool lexigraphicLess( const Vector3i& a, const Vector3i& b )
{
    if( a.x < b.x )
    {
        return true;
    }
    else if( a.x > b.x )
    {
        return false;
    }
    else
    {
        return lexigraphicLess( a.yz, b.yz );
    }
}

inline bool lexigraphicLess( const Vector4i& a, const Vector4i& b )
{
    if( a.x < b.x )
    {
        return true;
    }
    else if( a.x > b.x )
    {
        return false;
    }
    else
    {
        return lexigraphicLess( a.yzw(), b.yzw() );
    }
}

} } // core, libcgt
