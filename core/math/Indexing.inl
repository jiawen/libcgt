namespace libcgt { namespace core {

inline void indexToSubscript2D( int index, int width, int& x, int& y )
{
    y = index / width;
    x = index - y * width;
}

inline void indexToSubscript2D( int index, const Vector2i& size,
    int& x, int& y )
{
    indexToSubscript2D( index, size.x, x, y );
}

inline Vector2i indexToSubscript2D( int index, int width )
{
    int x;
    int y;
    indexToSubscript2D( index, width, x, y );
    return{ x, y };
}

inline Vector2i indexToSubscript2D( int index, const Vector2i& size )
{
    return indexToSubscript2D( index, size.x );
}

inline void indexToSubscript3D( int index, int width, int height,
    int& x, int& y, int& z )
{
    int wh = width * height;
    z = index / wh;

    int ky = index - z * wh;
    y = ky / width;

    x = ky - y * width;
}

inline void indexToSubscript3D( int index, const Vector3i& size,
    int& x, int&y, int& z )
{
    indexToSubscript3D( index, size.x, size.y, x, y, z );
}

inline Vector3i indexToSubscript3D( int index, int width, int height )
{
    int x;
    int y;
    int z;
    indexToSubscript3D( index, width, height, x, y, z );
    return{ x, y, z };
}

inline Vector3i indexToSubscript3D( int index, const Vector3i& size )
{
    return indexToSubscript3D( index, size.x, size.y );
}

inline int subscript2DToIndex( int x, int y, int width )
{
    return( y * width + x );
}

inline int subscript2DToIndex( const Vector2i& xy, int width )
{
    return subscript2DToIndex( xy.x, xy.y, width );
}

inline int subscript2DToIndex( const Vector2i& xy, const Vector2i& size )
{
    return subscript2DToIndex( xy.x, xy.y, size.x );
}

inline int subscript3DToIndex( int x, int y, int z, int width, int height )
{
    return( z * width * height + y * width + x );
}

inline int subscript3DToIndex( const Vector3i& xy, int width, int height )
{
    return subscript3DToIndex( xy.x, xy.y, xy.z, width, height );
}

inline int subscript3DToIndex( const Vector3i& xy, const Vector3i& size )
{
    return subscript3DToIndex( xy.x, xy.y, xy.z, size.x, size.y );
}

} } // core, libcgt
