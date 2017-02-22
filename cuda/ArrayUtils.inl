namespace libcgt { namespace cuda {

template< typename T >
bool saveTXT( const DeviceArray1D< T >& src, const std::string& filename )
{
    Array1D< T > h_array( src.size() );
    copy( src, h_array.writeView() );
    return saveTXT( h_array, filename );
}

template< typename T >
bool saveTXT( const DeviceArray2D< T >& src, const std::string& filename )
{
    Array2D< T > h_array( src.size() );
    copy( src, h_array.writeView() );
    return saveTXT( h_array, filename );
}

template< typename T >
bool saveTXT( const DeviceArray3D< T >& src, const std::string& filename )
{
    Array3D< T > h_array( src.size() );
    copy( src, h_array.writeView() );
    return saveTXT( h_array, filename );
}

} } // cuda, libcgt
