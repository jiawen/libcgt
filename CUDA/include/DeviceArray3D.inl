#if 0
#include "ThreadMath.cuh"
#include "MathUtils.h"

template< typename T >
__global__
void fillKernel( KernelArray3D< T > array, T value )
{
	int2 xy = libcgt::cuda::threadSubscript2DGlobal();
	if( !( libcgt::cuda::inRectangle( xy, make_int2( array.width, array.height ) ) ) )
	{
		return;
	}

	for( int z = 0; z < array.depth; ++z )
	{
		array( xy.x, xy.y, z ) = value;
	}
}
#endif

template< typename T >
DeviceArray3D< T >::DeviceArray3D() :

	m_width( -1 ),
	m_height( -1 ),
	m_depth( -1 ),

	m_sizeInBytes( 0 )

{
	m_pitchedPointer.ptr = NULL;
	m_pitchedPointer.pitch = 0;
	m_pitchedPointer.xsize = 0;
	m_pitchedPointer.ysize = 0;

	m_extent = make_cudaExtent( 0, 0, 0 );
}

template< typename T >
DeviceArray3D< T >::DeviceArray3D( int width, int height, int depth ) :

	m_width( -1 ),
	m_height( -1 ),
	m_depth( -1 ),

	m_sizeInBytes( 0 )

{
	m_pitchedPointer.ptr = NULL;
	m_pitchedPointer.pitch = 0;
	m_pitchedPointer.xsize = 0;
	m_pitchedPointer.ysize = 0;

	m_extent = make_cudaExtent( 0, 0, 0 );

	resize( width, height, depth );
}

template< typename T >
DeviceArray3D< T >::DeviceArray3D( const Array3D< T >& src ) :

	m_width( -1 ),
	m_height( -1 ),
	m_depth( -1 ),

	m_sizeInBytes( 0 )	

{
	m_pitchedPointer.ptr = NULL;
	m_pitchedPointer.pitch = 0;
	m_pitchedPointer.xsize = 0;
	m_pitchedPointer.ysize = 0;

	m_extent = make_cudaExtent( 0, 0, 0 );

	resize( src.width(), src.height(), src.depth() );
	copyFromHost( src );
}

template< typename T >
// virtual
DeviceArray3D< T >::~DeviceArray3D()
{
	destroy();
}

template< typename T >
bool DeviceArray3D< T >::isNull() const
{
	return( m_pitchedPointer.ptr == NULL );
}

template< typename T >
bool DeviceArray3D< T >::notNull() const
{
	return( m_pitchedPointer.ptr != NULL );
}

template< typename T >
int DeviceArray3D< T >::width() const
{
	return m_width;
}

template< typename T >
int DeviceArray3D< T >::height() const
{
	return m_height;
}

template< typename T >
int DeviceArray3D< T >::depth() const
{
	return m_depth;
}

template< typename T >
int3 DeviceArray3D< T >::size() const
{
	return make_int3( m_width, m_height, m_depth );
}

template< typename T >
int DeviceArray3D< T >::numElements() const
{
	return m_width * m_height;
}

template< typename T >
int DeviceArray3D< T >::subscriptToIndex( int x, int y, int z ) const
{
	return z * m_width * m_height + y * m_width + x;
}

template< typename T >
int3 DeviceArray3D< T >::indexToSubscript( int k ) const
{
	int wh = m_width * m_height;
	int z = k / wh;

	int ky = k - z * wh;
	int y = ky / m_width;

	int x = ky - y * m_width;
	return make_int3( x, y, z );
}

template< typename T >
size_t DeviceArray3D< T >::rowPitch() const
{
	return m_pitchedPointer.pitch;
}

template< typename T >
size_t DeviceArray3D< T >::slicePitch() const
{
	return m_pitchedPointer.pitch * m_height;
}

template< typename T >
size_t DeviceArray3D< T >::sizeInBytes() const
{
	return m_sizeInBytes;
}

template< typename T >
void DeviceArray3D< T >::resize( int width, int height, int depth )
{
	if( width == m_width &&
		height == m_height &&
		depth == m_depth )
	{
		return;
	}

	destroy();

	m_width = width;
	m_height = height;
	m_depth = depth;
	m_extent = make_cudaExtent( width * sizeof( T ), height, depth );

	CUDA_SAFE_CALL
	(
		cudaMalloc3D( &m_pitchedPointer, m_extent )
	);

	m_sizeInBytes = m_pitchedPointer.pitch * height * depth;
}

template< typename T >
void DeviceArray3D< T >::clear()
{
	CUDA_SAFE_CALL( cudaMemset3D( m_pitchedPointer, 0, m_extent ) );
}

template< typename T >
void DeviceArray3D< T >::fill( const T& value )
{
	// TODO: use a kernel?

	Array3D< T > h_array( width(), height(), depth(), value );
	copyFromHost( h_array );

#if 0
	// TODO: this is stupid, can also memcpy from an array...

	// TODO: tune the sizes
	dim3 block( 16, 16 );
	dim3 grid = libcgt::cuda::numBins2D( width(), height(), block );

	fillKernel< T > <<< grid, block >>>
	(
		kernelArray(), value
	);
	CUT_CHECK_ERROR( "DeviceArray3D< T >::fill() kernel launch\n" );
#endif
}

template< typename T >
void DeviceArray3D< T >::copyFromHost( const Array3D< T >& src )
{
	resize( src.width(), src.height(), src.depth() );

	cudaMemcpy3DParms params;

	params.kind = cudaMemcpyHostToDevice;

	// Since the source (on the host) is not pitched
	// make a pitchedPointer for it
	const T* srcPointer = src; // using const_cast since CUDA is stupid and wants a void*
	params.srcPtr = make_cudaPitchedPtr( const_cast< T* >( srcPointer ), src.width() * sizeof( T ), src.width(), src.height() );
	params.srcArray = NULL; // we're not copying a CUDA array
	params.srcPos = make_cudaPos( 0, 0, 0 );
	
	params.dstPtr = m_pitchedPointer;
	params.dstArray = NULL; // we're not copying a CUDA array
	params.dstPos = make_cudaPos( 0, 0, 0 );	

	params.extent = m_extent;	

	CUDA_SAFE_CALL( cudaMemcpy3D( &params ) );
}

template< typename T >
void DeviceArray3D< T >::copyToHost( Array3D< T >& dst ) const
{
	dst.resize( width(), height(), depth() );

	cudaMemcpy3DParms params;

	params.kind = cudaMemcpyDeviceToHost;

	params.srcPtr = m_pitchedPointer;
	params.srcArray = NULL; // we're not copying a CUDA array
	params.srcPos = make_cudaPos( 0, 0, 0 );
	
	// Since the destination (on the host) is not pitched
	// make a pitchedPointer for it
	params.dstPtr = make_cudaPitchedPtr( dst, dst.width() * sizeof( T ), dst.width(), dst.height() );
	params.dstArray = NULL; // we're not copying a CUDA array
	params.dstPos = make_cudaPos( 0, 0, 0 );
	
	params.extent = m_extent;

	CUDA_SAFE_CALL( cudaMemcpy3D( &params ) );
}

template< typename T >
DeviceArray3D< T >::operator cudaPitchedPtr() const
{
	return m_pitchedPointer;
}

template< typename T >
cudaPitchedPtr DeviceArray3D< T >::pitchedPointer() const
{
	return m_pitchedPointer;
}

template< typename T >
KernelArray3D< T > DeviceArray3D< T >::kernelArray() const
{
	return KernelArray3D< T >( m_pitchedPointer, m_width, m_height, m_depth );
}

template< typename T >
void DeviceArray3D< T >::load( const char* filename )
{
	Array3D< T > h_arr( filename );
	if( !( h_arr.isNull() ) )
	{
		resize( h_arr.width(), h_arr.height(), h_arr.depth() );
		copyFromHost( h_arr );
	}
}

template< typename T >
void DeviceArray3D< T >::save( const char* filename ) const
{
	Array3D< T > h_arr( width(), height(), height() );
	copyToHost( h_arr );
	h_arr.save( filename );
}

template< typename T >
void DeviceArray3D< T >::destroy()
{
	if( notNull() )
	{
		CUDA_SAFE_CALL( cudaFree( m_pitchedPointer.ptr ) );
		m_pitchedPointer.ptr = NULL;
		m_pitchedPointer.pitch = 0;
		m_pitchedPointer.xsize = 0;
		m_pitchedPointer.ysize = 0;
	}

	m_width = -1;
	m_height = -1;
	m_depth = -1;

	m_sizeInBytes = 0;

	m_extent = make_cudaExtent( 0, 0, 0 );
}
