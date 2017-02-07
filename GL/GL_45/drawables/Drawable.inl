// TODO(jiawen): move this into GLVertexBufferObject itself.
// Maps the range covered by attribute i for writing.
// You may only map one attribute at a time.
// TODO(jiawen): can you map more than one range at a time?
template< typename T >
GLDrawable::MappedBuffer< T > GLDrawable::mapAttribute( int i )
{
    GLDrawable* drawable = nullptr;
    const PlanarVertexBufferCalculator::AttributeInfo& info =
        m_calculator.getAttributeInfo( i );
    Array1DWriteView< T > buffer;

    // Check that the i-th attribute has the right size.
    if( info.vertexStride == sizeof( T ) )
    {
        buffer = m_vbo.mapRangeAs< T >
        (
            info.offset, info.arraySize,
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
