template< typename T >
__inline__ __host__ __device__
KernelArray1D< T >::KernelArray1D(
    typename KernelArray1D< T >::VoidPointer pointer, size_t size,
    ptrdiff_t stride ) :
    md_pointer( reinterpret_cast< KernelArray1D::UInt8Pointer >( pointer ) ),
    m_size( size ),
    m_stride( stride )
{

}

template< typename T >
__inline__ __device__
size_t KernelArray1D< T >::width() const
{
    return m_size;
}

template< typename T >
__inline__ __device__
size_t KernelArray1D< T >::size() const
{
    return m_size;
}

template< typename T >
__inline__ __device__
ptrdiff_t KernelArray1D< T >::stride() const
{
    return m_stride;
}

template< typename T >
__inline__ __device__
T* KernelArray1D< T >::pointer() const
{
    return reinterpret_cast< T* >( md_pointer );
}

template< typename T >
__inline__ __device__
T* KernelArray1D< T >::elementPointer( size_t x ) const
{
    return reinterpret_cast< T* >( md_pointer + x * m_stride );
}

template< typename T >
__inline__ __device__
T& KernelArray1D< T >::operator [] ( size_t i ) const
{
    return *elementPointer( i );
}
