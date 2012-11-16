#pragma once

#include <vector>

#include "DevicePool.h"

template< typename UsedListTag >
class HostPool
{
public:

	// creates a null pool
	HostPool();

	// creates a device memory pool of capacity elements,
	// where each element occupies elementSizeBytes bytes
	//
	// capacity must be a power of two
	HostPool( int capacity, int elementSizeBytes );
	virtual ~HostPool();

	bool isNull() const;
	bool notNull() const;

	// returns the number of elements
	int capacity() const;

	// return the size of an element, in bytes
	int elementSizeBytes() const;

	// returns the total size of the pool, in bytes, including overhead
	size_t sizeInBytes() const;

	// returns the number of free elements
	// (potentially expensive - incurs a GPU copy)
	int numFreeElements();

	// resizes the pool to a new capacity, clearing it at the same time
	void resize( int capacity, int elementSizeBytes );

	// clears the pool: marks all elements as free
	// does *not* touch the data
	void clear();

	template< typename T >
	__inline__ __device__
	T* getElement( int index );

	void copyFromDevice( const DevicePool< UsedListTag >& pool );

	//private:

	int m_capacity;
	int m_elementSizeBytes;

	// stores a list of free (still-available) element (not byte) indices in m_backingStore
	// initially free
	std::vector< int > m_freeList;

	// stores a list of allocated (not currently available) element indices in m_backingStore
	// if the value is positive, then it's still in use
	// if the value is negative, then it's been marked as freed and will be collected in the next call to collect()
	std::vector< UsedListEntry< UsedListTag > > m_usedList;

	// temporary storage for the garbage collector
	std::vector< UsedListEntry< UsedListTag > > m_collectedList;

	// backing store of capacity * elementSizeBytes bytes
	std::vector< ubyte > m_backingStore;
};

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
__inline__ __device__
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

	pool.md_freeList.copyToHost( m_freeList );
	pool.md_usedList.copyToHost( m_usedList );
	pool.md_collectedList.copyToHost( m_collectedList );

	pool.md_backingStore.copyToHost( m_backingStore );
}
