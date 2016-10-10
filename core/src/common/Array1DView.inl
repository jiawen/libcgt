template< typename T >
Array1DView< T >::Array1DView( typename Array1DView< T >::VoidPointer pointer,
    size_t size ) :
    m_size( size ),
    m_stride( sizeof( T ) ),
    m_pointer( reinterpret_cast< typename Array1DView< T >::UInt8Pointer >(
        pointer ) )
{

}

template< typename T >
Array1DView< T >::Array1DView( typename Array1DView< T >::VoidPointer pointer,
    size_t size, std::ptrdiff_t stride ) :
    m_size( size ),
    m_stride( stride ),
    m_pointer( reinterpret_cast< typename Array1DView< T >::UInt8Pointer >(
        pointer ) )
{

}

template< typename T >
bool Array1DView< T >::isNull() const
{
    return( m_pointer == nullptr );
}

template< typename T >
bool Array1DView< T >::notNull() const
{
    return( m_pointer != nullptr );
}

template< typename T >
void Array1DView< T >::setNull()
{
    m_size = 0;
    m_stride = 0;
    m_pointer = nullptr;
}

template< typename T >
template< typename U, typename >
Array1DView< T >::operator const T* () const
{
    return reinterpret_cast< const T* >( m_pointer );
}

template< typename T >
Array1DView< T >::operator T* ()
{
    return reinterpret_cast< T* >( m_pointer );
}

template< typename T >
template< typename U, typename >
const T* Array1DView< T >::pointer() const
{
    return reinterpret_cast< const T* >( m_pointer );
}

template< typename T >
T* Array1DView< T >::pointer()
{
    return reinterpret_cast< T* >( m_pointer );
}

template< typename T >
T* Array1DView< T >::elementPointer( size_t x )
{
    return reinterpret_cast< T* >( m_pointer + x * elementStrideBytes() );
}

template< typename T >
T& Array1DView< T >::operator [] ( size_t k )
{
    T* q = reinterpret_cast< T* >( m_pointer + k * m_stride );
    return *q;
}

template< typename T >
size_t Array1DView< T >::width() const
{
    return m_size;
}

template< typename T >
size_t Array1DView< T >::size() const
{
    return m_size;
}

template< typename T >
size_t Array1DView< T >::numElements() const
{
    return m_size;
}

template< typename T >
std::ptrdiff_t Array1DView< T >::elementStrideBytes() const
{
    return m_stride;
}

template< typename T >
std::ptrdiff_t Array1DView< T >::stride() const
{
    return m_stride;
}

template< typename T >
bool Array1DView< T >::elementsArePacked() const
{
    return( m_stride == sizeof( T ) );
}

template< typename T >
bool Array1DView< T >::packed() const
{
    return( m_stride == sizeof( T ) );
}

template< typename T >
template< typename U, typename >
Array1DView< T >::operator Array1DView< const T >() const
{
    return Array1DView< const T >( m_pointer, m_size, m_stride );
}
