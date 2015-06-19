#include "HostPool.h"

// STL
#include <numeric> // for iota

#include <common/ArrayUtils.h>

HostPool::HostPool() :

	m_capacity( -1 ),
	m_elementSizeBytes( -1 )

{

}

HostPool::HostPool( int capacity, int elementSizeBytes )
{
	resize( capacity, elementSizeBytes );
}

// virtual
HostPool::~HostPool()
{

}

bool HostPool::isNull() const
{
	return( m_freeList.size() > 0 && m_backingStore.size() > 0 );
}

bool HostPool::notNull() const
{
	return !isNull();
}

int HostPool::capacity() const
{
	return m_capacity;
}

int HostPool::elementSizeBytes() const
{
	return m_elementSizeBytes;
};

size_t HostPool::sizeInBytes() const
{
	size_t esb = m_elementSizeBytes;
	size_t poolSizeBytes = esb * capacity();

	return poolSizeBytes + m_freeList.size() * sizeof( int );		
}

int HostPool::numFreeElements()
{
	return static_cast< int >( m_freeList.size() );
}

void HostPool::resize( int capacity, int elementSizeBytes )
{
	m_capacity = capacity;
	m_elementSizeBytes = elementSizeBytes;
	m_freeList.resize( capacity );
	m_backingStore.resize( capacity * elementSizeBytes );

	clear();	
}

void HostPool::clear()
{
	// generate free list: [0,capacity)
	m_freeList.resize( m_capacity );
	std::iota( m_freeList.begin(), m_freeList.end(), 0 );
}

void HostPool::copyFromDevice( const DevicePool& pool )
{
	m_capacity = pool.capacity();
	m_elementSizeBytes = pool.elementSizeBytes();

	printf( "Copying free list...\n" );
	pool.md_freeList.copyToHost( m_freeList );

	printf( "Copying backing store...\n" );
	pool.md_backingStore.copyToHost( m_backingStore );
}

void HostPool::copyToDevice( DevicePool& pool )
{
	pool.resize( m_capacity, m_elementSizeBytes );

	pool.md_freeList.copyFromHost( m_freeList );
	pool.md_backingStore.copyFromHost( m_backingStore );
}

void HostPool::loadBinary( FILE* fp )
{
	// TODO: error checking

	fread( &m_capacity, sizeof( int ), 1, fp );
	fread( &m_elementSizeBytes, sizeof( int ), 1, fp );
	ArrayUtils::loadBinary( fp, m_freeList );
	ArrayUtils::loadBinary( fp, m_backingStore );
}

void HostPool::saveBinary( FILE* fp )
{
	// TODO: error checking

	fwrite( &m_capacity, sizeof( int ), 1, fp );
	fwrite( &m_elementSizeBytes, sizeof( int ), 1, fp );
	ArrayUtils::saveBinary( m_freeList, fp );
	ArrayUtils::saveBinary( m_backingStore, fp );
}
