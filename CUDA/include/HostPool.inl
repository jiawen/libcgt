template< typename UsedListTag >
__inline__ __host__
HostPool< UsedListTag >::HostPool() :

	m_capacity( -1 ),
	m_elementSizeBytes( -1 )

{

}

template< typename UsedListTag >
__inline__ __host__
HostPool< UsedListTag >::HostPool( int capacity, int elementSizeBytes )
{
	resize( capacity, elementSizeBytes );
}

// virtual
template< typename UsedListTag >
__inline__ __host__
HostPool< UsedListTag >::~HostPool()
{

}

template< typename UsedListTag >
__inline__ __host__
bool HostPool< UsedListTag >::isNull() const
{
	return( m_freeList.size() > 0 && m_backingStore.size() > 0 );
}

template< typename UsedListTag >
__inline__ __host__
bool HostPool< UsedListTag >::notNull() const
{
	return !isNull();
}

template< typename UsedListTag >
__inline__ __host__
int HostPool< UsedListTag >::capacity() const
{
	return m_capacity;
}

template< typename UsedListTag >
__inline__ __host__
int HostPool< UsedListTag >::elementSizeBytes() const
{
	return m_elementSizeBytes;
};

template< typename UsedListTag >
__inline__ __host__
size_t HostPool< UsedListTag >::sizeInBytes() const
{
	size_t esb = m_elementSizeBytes;
	size_t poolSizeBytes = esb * capacity();

	return poolSizeBytes +
		m_freeList.size() * sizeof( int ) +
		m_usedList.size() * sizeof( UsedListEntry< UsedListTag > ) +
		m_collectedList.size() * sizeof( UsedListEntry< UsedListTag > );
}


template< typename UsedListTag >
__inline__ __host__
int HostPool< UsedListTag >::numFreeElements()
{
	exit( -1 );
	return m_freeList.size();
}

template< typename UsedListTag >
__inline__ __host__
void HostPool< UsedListTag >::resize( int capacity, int elementSizeBytes )
{
	m_capacity = capacity;
	m_elementSizeBytes = elementSizeBytes;
	m_freeList.resize( capacity );
	m_usedList.resize( capacity );
	m_collectedList.resize( capacity );
	m_backingStore.resize( capacity * elementSizeBytes );

	clear();	
}

template< typename UsedListTag >
__inline__ __host__
void HostPool< UsedListTag >::clear()
{
	// generate free list: [0,capacity)
	m_freeList.resize( m_capacity );
	std::iota( m_freeList.begin(), m_freeList.end(), 0 );
}

template< typename UsedListTag >
template< typename T >
__inline__ __host__
T* HostPool< UsedListTag >::getElement( int index )
{
	ubyte* pElementStart = &( m_backingStore[ index * m_elementSizeBytes ] );
	return reinterpret_cast< T* >( pElementStart );
}

template< typename UsedListTag >
__inline__ __host__
void HostPool< UsedListTag >::copyFromDevice( const DevicePool< UsedListTag >& pool )
{
	m_capacity = pool.capacity();
	m_elementSizeBytes = pool.elementSizeBytes();

	printf( "Copying free list...\n" );
	pool.md_freeList.copyToHost( m_freeList );
	printf( "Copying used list...\n" );
	pool.md_usedList.copyToHost( m_usedList );
	printf( "Copying collected list...\n" );
	pool.md_collectedList.copyToHost( m_collectedList );

	printf( "Copying backing store...\n" );
	pool.md_backingStore.copyToHost( m_backingStore );
}

template< typename UsedListTag >
__inline__ __host__
void HostPool< UsedListTag >::copyToDevice( DevicePool< UsedListTag >& pool )
{
	pool.resize( m_capacity, m_elementSizeBytes );

	pool.md_freeList.copyFromHost( m_freeList );
	pool.md_usedList.copyFromHost( m_usedList );
	pool.md_collectedList.copyFromHost( m_collectedList );
	pool.md_backingStore.copyFromHost( m_backingStore );
}

template< typename UsedListTag >
__inline__ __host__
void HostPool< UsedListTag >::loadBinary( FILE* fp )
{
	// TODO: error checking

	fread( &m_capacity, sizeof( int ), 1, fp );
	fread( &m_elementSizeBytes, sizeof( int ), 1, fp );
	ArrayUtils::loadBinary( fp, m_freeList );
	ArrayUtils::loadBinary( fp, m_usedList );
	ArrayUtils::loadBinary( fp, m_collectedList );
	ArrayUtils::loadBinary( fp, m_backingStore );
}


template< typename UsedListTag >
__inline__ __host__
void HostPool< UsedListTag >::saveBinary( FILE* fp )
{
	// TODO: error checking

	fwrite( &m_capacity, sizeof( int ), 1, fp );
	fwrite( &m_elementSizeBytes, sizeof( int ), 1, fp );
	ArrayUtils::saveBinary( m_freeList, fp );
	ArrayUtils::saveBinary( m_usedList, fp );
	ArrayUtils::saveBinary( m_collectedList, fp );
	ArrayUtils::saveBinary( m_backingStore, fp );
}
