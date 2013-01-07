#include <numeric>

__inline__ __host__
HostPool::HostPool() :

	m_capacity( -1 ),
	m_elementSizeBytes( -1 )

{

}

__inline__ __host__
HostPool::HostPool( int capacity, int elementSizeBytes )
{
	resize( capacity, elementSizeBytes );
}

// virtual
__inline__ __host__
HostPool::~HostPool()
{

}

__inline__ __host__
bool HostPool::isNull() const
{
	return( m_freeList.size() > 0 && m_backingStore.size() > 0 );
}

__inline__ __host__
bool HostPool::notNull() const
{
	return !isNull();
}

__inline__ __host__
int HostPool::capacity() const
{
	return m_capacity;
}

__inline__ __host__
int HostPool::elementSizeBytes() const
{
	return m_elementSizeBytes;
};

__inline__ __host__
size_t HostPool::sizeInBytes() const
{
	size_t esb = m_elementSizeBytes;
	size_t poolSizeBytes = esb * capacity();

	return poolSizeBytes + m_freeList.size() * sizeof( int );		
}

__inline__ __host__
int HostPool::numFreeElements()
{
	return m_freeList.size();
}

__inline__ __host__
void HostPool::resize( int capacity, int elementSizeBytes )
{
	m_capacity = capacity;
	m_elementSizeBytes = elementSizeBytes;
	m_freeList.resize( capacity );
	m_backingStore.resize( capacity * elementSizeBytes );

	clear();	
}

__inline__ __host__
void HostPool::clear()
{
	// generate free list: [0,capacity)
	m_freeList.resize( m_capacity );
	std::iota( m_freeList.begin(), m_freeList.end(), 0 );
}

template< typename T >
__inline__ __host__
T* HostPool::getElement( int index )
{
	ubyte* pElementStart = &( m_backingStore[ index * m_elementSizeBytes ] );
	return reinterpret_cast< T* >( pElementStart );
}

__inline__ __host__
void HostPool::copyFromDevice( const DevicePool& pool )
{
	m_capacity = pool.capacity();
	m_elementSizeBytes = pool.elementSizeBytes();

	printf( "Copying free list...\n" );
	pool.md_freeList.copyToHost( m_freeList );

	printf( "Copying backing store...\n" );
	pool.md_backingStore.copyToHost( m_backingStore );
}

__inline__ __host__
void HostPool::copyToDevice( DevicePool& pool )
{
	pool.resize( m_capacity, m_elementSizeBytes );

	pool.md_freeList.copyFromHost( m_freeList );
	pool.md_backingStore.copyFromHost( m_backingStore );
}

__inline__ __host__
void HostPool::loadBinary( FILE* fp )
{
	// TODO: error checking

	fread( &m_capacity, sizeof( int ), 1, fp );
	fread( &m_elementSizeBytes, sizeof( int ), 1, fp );
	ArrayUtils::loadBinary( fp, m_freeList );
	ArrayUtils::loadBinary( fp, m_backingStore );
}

__inline__ __host__
void HostPool::saveBinary( FILE* fp )
{
	// TODO: error checking

	fwrite( &m_capacity, sizeof( int ), 1, fp );
	fwrite( &m_elementSizeBytes, sizeof( int ), 1, fp );
	ArrayUtils::saveBinary( m_freeList, fp );
	ArrayUtils::saveBinary( m_backingStore, fp );
}
