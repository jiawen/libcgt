namespace libcgt { namespace core {

template< typename Func >
inline void for2D( const Vector2i& count, Func func )
{
    for2D( Vector2i( 0, 0 ), count, Vector2i( 1, 1 ), func );
}

template< typename Func >
inline void for2D( const Vector2i& count,
    const std::string& progressPrefix,
    Func func )
{
    for2D( Vector2i( 0, 0 ), count, Vector2i( 1, 1 ), progressPrefix,
        func );
}

template< typename Func >
inline void for2D( const Vector2i& first, const Vector2i& count,
    Func func )
{
    for2D( first, count, { 1, 1 }, func );
}

template< typename Func >
inline void for2D( const Vector2i& first, const Vector2i& count,
    const Vector2i& step, Func func )
{
    for( int j = 0; j < count.y; j += step.y )
    {
        int y = first.y + j;
        for( int i = 0; i < count.x; i += step.x )
        {
            int x = first.x + i;

            func( { x, y } );
        }
    }
}

template< typename Func >
inline void for2D( const Vector2i& first, const Vector2i& count,
    const Vector2i& step,
    const std::string& progressPrefix,
    Func func )
{
    ProgressReporter pr( progressPrefix, count.y / step.y );

    for( int j = 0; j < count.y; j += step.y )
    {
        int y = first.y + j;
        for( int i = 0; i < count.x; i += step.x )
        {
            int x = first.x + i;

            func( { x, y } );
        }
        pr.notifyAndPrintProgressString();
    }
}

template< typename Func >
inline void for3D( const Vector3i& count, Func func )
{
    for3D( { 0, 0, 0 }, count, { 1, 1, 1 }, func );
}

template< typename Func >
inline void for3D( const Vector3i& count,
    const std::string& progressPrefix,
    Func func )
{
    for3D( { 0, 0, 0 }, count, { 1, 1, 1 }, progressPrefix, func );
}

template< typename Func >
inline void for3D( const Vector3i& first, const Vector3i& count,
    Func func )
{
    for3D( first, count, { 1, 1, 1 }, func );
}

template< typename Func >
inline void for3D( const Vector3i& first, const Vector3i& count,
    const Vector3i& step, Func func )
{
    for( int k = 0; k < count.z; k += step.z )
    {
        int z = first.z + k;
        for( int j = 0; j < count.y; j += step.y )
        {
            int y = first.y + j;
            for( int i = 0; i < count.x; i += step.x )
            {
                int x = first.x + i;

                func( { x, y, z } );
            }
        }
    }
}

template< typename Func >
inline void for3D( const Vector3i& first, const Vector3i& count,
    const Vector3i& step,
    const std::string& progressPrefix,
    Func func )
{
    ProgressReporter pr( progressPrefix, count.z / step.z );

    for( int k = 0; k < count.z; k += step.z )
    {
        int z = first.z + k;
        for( int j = 0; j < count.y; j += step.y )
        {
            int y = first.y + j;
            for( int i = 0; i < count.x; i += step.x )
            {
                int x = first.x + i;

                func( { x, y, z } );
            }
        }
        pr.notifyAndPrintProgressString();
    }
}

} } // core, libcgt
