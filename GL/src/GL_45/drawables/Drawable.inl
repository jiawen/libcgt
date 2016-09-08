// TODO(jiawen): move this into GLVertexBufferObject itself.
// Maps the range covered by attribute i for writing.
// You may only map one attribute at a time.
// TODO(jiawen): can you map more than one range at a time?
template< typename T >
GLDrawable::MappedBuffer< T > GLDrawable::mapAttribute( int i )
{
    GLDrawable* drawable = nullptr;
    Array1DView< T > buffer;
    if( m_calculator.vertexSizeOf( i ) == sizeof( T ) )
    {
        buffer = m_vbo.mapRangeAs< T >
            (
            m_calculator.offsetOf( i ),
            m_calculator.arraySizeOf( i ),
            GLBufferObject::MapRangeAccess::WRITE_BIT |
            GLBufferObject::MapRangeAccess::INVALIDATE_RANGE_BIT
            );
        if( buffer.notNull() )
        {
            drawable = this;
        }
    }
    return MappedBuffer< T >( drawable, buffer );
}
