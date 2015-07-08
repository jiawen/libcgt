template< typename T >
__inline__ __device__ __host__
KernelArray3D< T >::KernelArray3D()
{

}

// A cudaPitchedPtr contains
// width: logical width of the array in elements
// xsize = elementSize * width, in bytes
// ysize = height of the array in elements
// pitch = roundUpToAlignment( xsize ), in bytes
template< typename T >
__inline__ __device__ __host__
KernelArray3D< T >::KernelArray3D( cudaPitchedPtr d_pitchedPointer, int depth ) :

    md_pPitchedPointer( reinterpret_cast< T* >( d_pitchedPointer.ptr ) ),
    m_width( static_cast< int >( d_pitchedPointer.xsize / sizeof( T ) ) ),
    m_height( d_pitchedPointer.ysize ),
    m_depth( depth ),
    m_rowPitch( d_pitchedPointer.pitch )

{

}

template< typename T >
__inline__ __device__ __host__
KernelArray3D< T >::KernelArray3D( T* d_pLinearPointer, int width, int height, int depth ) :

    md_pPitchedPointer( d_pLinearPointer ),
    m_width( width ),
    m_height( height ),
    m_depth( depth ),
    m_rowPitch( width * sizeof( T ) )

{

}

template< typename T >
__inline__ __device__ __host__
KernelArray3D< T >::KernelArray3D( T* d_pLinearPointer, const int3& size ) :

    md_pPitchedPointer( d_pLinearPointer ),
    m_width( size.x ),
    m_height( size.y ),
    m_depth( size.z ),
    m_rowPitch( size.x * sizeof( T ) )

{

}

template< typename T >
__inline__ __device__
T* KernelArray3D< T >::rowPointer( int y, int z )
{
    uint8_t* p = reinterpret_cast< uint8_t* >( md_pPitchedPointer );
    uint8_t* pSlice = p + z * slicePitch();
    return reinterpret_cast< T* >( pSlice + y * m_rowPitch );
}

template< typename T >
__inline__ __device__
const T* KernelArray3D< T >::rowPointer( int y, int z ) const
{
    uint8_t* p = reinterpret_cast< uint8_t* >( md_pPitchedPointer );
    uint8_t* pSlice = p + z * slicePitch();
    return reinterpret_cast< T* >( pSlice + y * m_rowPitch );
}

template< typename T >
__inline__ __device__
T* KernelArray3D< T >::slicePointer( int z )
{
    uint8_t* p = reinterpret_cast< uint8_t* >( md_pPitchedPointer );
    return reinterpret_cast< T* >( p + z * slicePitch() );
}

template< typename T >
__inline__ __device__
int KernelArray3D< T >::width() const
{
    return m_width;
}

template< typename T >
__inline__ __device__
int KernelArray3D< T >::height() const
{
    return m_height;
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
    return m_rowPitch;
}

template< typename T >
__inline__ __device__
size_t KernelArray3D< T >::slicePitch() const
{
    return m_rowPitch * m_height;
}

template< typename T >
__inline__ __device__
const T* KernelArray3D< T >::slicePointer( int z ) const
{
    uint8_t* p = reinterpret_cast< uint8_t* >( md_pPitchedPointer );
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
T& KernelArray3D< T >::operator [] ( const int3& xyz )
{
    return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}

template< typename T >
__inline__ __device__
const T& KernelArray3D< T >::operator [] ( const int3& xyz ) const
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
        reinterpret_cast< S* >( md_pPitchedPointer ), outputWidth, outputHeight, outputDepth
    );
}

template< typename T >
template< typename S >
__inline__ __device__ __host__
KernelArray3D< S > KernelArray3D< T >::reinterpretAs( const int3& outputSize )
{
    return reinterpretAs< S >( outputSize.x, outputSize.y, outputSize.z );
}
