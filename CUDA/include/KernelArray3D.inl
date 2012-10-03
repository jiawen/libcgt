template< typename T >
KernelArray3D< T >::KernelArray3D()
{

}
	
template< typename T >
KernelArray3D< T >::KernelArray3D( cudaPitchedPtr d_pitchedPointer, int _width, int _height, int _depth ) :

	pitchedPointer( d_pitchedPointer ),
	width( _width ),
	height( _height ),
	depth( _depth ),
	slicePitch( d_pitchedPointer.pitch * d_pitchedPointer.ysize )

{

}

template< typename T >
KernelArray3D< T >::KernelArray3D( T* d_pLinearPointer, int _width, int _height, int _depth ) :

	width( _width ),
	height( _height ),
	depth( _depth ),
	slicePitch( _width * sizeof( T ) * _height )

{
	pitchedPointer.ptr = d_pLinearPointer;
	pitchedPointer.pitch = _width * sizeof( T );
	pitchedPointer.xsize = _width;
	pitchedPointer.ysize = _height;
}

template< typename T >
KernelArray3D< T >::KernelArray3D( T* d_pLinearPointer, const int3& size ) :

	width( size.x ),
	height( size.y ),
	depth( size.z ),
	slicePitch( size.x * sizeof( T ) * size.y )

{
	pitchedPointer.ptr = d_pLinearPointer;
	pitchedPointer.pitch = size.x * sizeof( T );
	pitchedPointer.xsize = size.x;
	pitchedPointer.ysize = size.y;
}

template< typename T >
T* KernelArray3D< T >::rowPointer( int y, int z )
{
	ubyte* p = reinterpret_cast< ubyte* >( pitchedPointer.ptr );

	size_t rowPitch = pitchedPointer.pitch;

	// TODO: switch pointer arithmetic to array indexing?
	ubyte* pSlice = p + z * slicePitch;
	return reinterpret_cast< T* >( pSlice + y * rowPitch );
}

template< typename T >
const T* KernelArray3D< T >::rowPointer( int y, int z ) const
{
	ubyte* p = reinterpret_cast< ubyte* >( pitchedPointer.ptr );

	size_t rowPitch = pitchedPointer.pitch;

	// TODO: switch pointer arithmetic to array indexing?
	ubyte* pSlice = p + z * slicePitch;
	return reinterpret_cast< T* >( pSlice + y * rowPitch );
}

template< typename T >
T* KernelArray3D< T >::slicePointer( int z )
{
	ubyte* p = reinterpret_cast< ubyte* >( pitchedPointer.ptr );

	// TODO: switch pointer arithmetic to array indexing?
	return reinterpret_cast< T* >( p + z * slicePitch );
}

template< typename T >
int3 KernelArray3D< T >::size() const
{
	return make_int3( width, height, depth );
}

template< typename T >
const T* KernelArray3D< T >::slicePointer( int z ) const
{
	char* p = reinterpret_cast< char* >( pitchedPointer.ptr );

	// TODO: switch pointer arithmetic to array indexing?
	return reinterpret_cast< T* >( p + z * slicePitch );
}

template< typename T >
const T& KernelArray3D< T >::operator () ( int x, int y, int z ) const
{
	return rowPointer( y, z )[ x ];
}

template< typename T >
T& KernelArray3D< T >::operator () ( int x, int y, int z )
{
	return rowPointer( y, z )[ x ];
}

template< typename T >
T& KernelArray3D< T >::operator () ( const int3& xyz )
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}

template< typename T >
const T& KernelArray3D< T >::operator () ( const int3& xyz ) const
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}
