template< typename T >
__inline__ __device__ __host__
KernelArray3D< T >::KernelArray3D()
{

}
	
template< typename T >
__inline__ __device__ __host__
KernelArray3D< T >::KernelArray3D( cudaPitchedPtr d_pitchedPointer, int width, int height, int depth ) :

	md_pitchedPointer( d_pitchedPointer ),
	m_depth( depth )

{

}

template< typename T >
__inline__ __device__ __host__
KernelArray3D< T >::KernelArray3D( T* d_pLinearPointer, int width, int height, int depth ) :

	m_depth( depth )

{
	md_pitchedPointer.ptr = d_pLinearPointer;
	md_pitchedPointer.pitch = width * sizeof( T );
	md_pitchedPointer.xsize = width * sizeof( T );
	md_pitchedPointer.ysize = height;
}

template< typename T >
__inline__ __device__ __host__
KernelArray3D< T >::KernelArray3D( T* d_pLinearPointer, const int3& size ) :

	m_depth( size.z )

{
	md_pitchedPointer.ptr = d_pLinearPointer;
	md_pitchedPointer.pitch = size.x * sizeof( T );
	md_pitchedPointer.xsize = size.x * sizeof( T );
	md_pitchedPointer.ysize = size.y;
}

template< typename T >
__inline__ __device__
T* KernelArray3D< T >::rowPointer( int y, int z )
{
	ubyte* p = reinterpret_cast< ubyte* >( md_pitchedPointer.ptr );

	size_t rowPitch = md_pitchedPointer.pitch;

	// TODO: switch pointer arithmetic to array indexing?
	ubyte* pSlice = p + z * slicePitch();
	return reinterpret_cast< T* >( pSlice + y * rowPitch );
}

template< typename T >
__inline__ __device__
const T* KernelArray3D< T >::rowPointer( int y, int z ) const
{
	ubyte* p = reinterpret_cast< ubyte* >( md_pitchedPointer.ptr );

	size_t rowPitch = md_pitchedPointer.pitch;

	// TODO: switch pointer arithmetic to array indexing?
	ubyte* pSlice = p + z * slicePitch();
	return reinterpret_cast< T* >( pSlice + y * rowPitch );
}

template< typename T >
__inline__ __device__
T* KernelArray3D< T >::slicePointer( int z )
{
	ubyte* p = reinterpret_cast< ubyte* >( md_pitchedPointer.ptr );

	// TODO: switch pointer arithmetic to array indexing?
	return reinterpret_cast< T* >( p + z * slicePitch() );
}

template< typename T >
__inline__ __device__
int KernelArray3D< T >::width() const
{
	return md_pitchedPointer.xsize / sizeof( T );
}

template< typename T >
__inline__ __device__
int KernelArray3D< T >::height() const
{
	return md_pitchedPointer.ysize;
}

template< typename T >
__inline__ __device__
int KernelArray3D< T >::depth() const
{
	return m_depth;
}

template< typename T >
__inline__ __device__
int3 KernelArray3D< T >::size() const
{
	return make_int3( width(), height(), depth() );
}

template< typename T >
__inline__ __device__
size_t KernelArray3D< T >::rowPitch() const
{
	return md_pitchedPointer.pitch;
}

template< typename T >
__inline__ __device__
size_t KernelArray3D< T >::slicePitch() const
{
	return md_pitchedPointer.pitch * md_pitchedPointer.ysize;
}

template< typename T >
__inline__ __device__
const T* KernelArray3D< T >::slicePointer( int z ) const
{
	char* p = reinterpret_cast< char* >( md_pitchedPointer.ptr );

	// TODO: switch pointer arithmetic to array indexing?
	return reinterpret_cast< T* >( p + z * slicePitch() );
}

template< typename T >
__inline__ __device__
const T& KernelArray3D< T >::operator () ( int x, int y, int z ) const
{
	return rowPointer( y, z )[ x ];
}

template< typename T >
__inline__ __device__
T& KernelArray3D< T >::operator () ( int x, int y, int z )
{
	return rowPointer( y, z )[ x ];
}

template< typename T >
__inline__ __device__
T& KernelArray3D< T >::operator () ( const int3& xyz )
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}

template< typename T >
__inline__ __device__
const T& KernelArray3D< T >::operator () ( const int3& xyz ) const
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}

template< typename T >
template< typename S >
__inline__ __device__ __host__
KernelArray3D< S > KernelArray3D< T >::reinterpretAs( int outputWidth, int outputHeight, int outputDepth )
{
	return KernelArray3D< S >
	(
		reinterpret_cast< S* >( md_pitchedPointer.ptr ), outputWidth, outputHeight, outputDepth
	);	
}

template< typename T >
template< typename S >
__inline__ __device__ __host__
KernelArray3D< S > KernelArray3D< T >::reinterpretAs( const int3& outputSize )
{
	return reinterpretAs< S >( outputSize.x, outputSize.y, outputSize.z );
}
