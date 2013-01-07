#pragma once

#include <cstdio>
#include <vector>

#include "DevicePool.h"

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
	__inline__ __host__
	T* getElement( int index );

	void copyFromDevice( const DevicePool& pool );
	void copyToDevice( DevicePool& pool );

	void loadBinary( FILE* fp );
	void saveBinary( FILE* fp );

	//private:

	int m_capacity;
	int m_elementSizeBytes;

	// stores a list of free (still-available) element (not byte) indices in md_backingStore
	// initially, every element is free
	// and so is an entirely full queue of [0 1 2 3 ... m_capacity)
	// head = 0, tail = m_capacity
	std::vector< int > m_freeList;

	// backing store of capacity * elementSizeBytes bytes
	std::vector< ubyte > m_backingStore;
};

template< typename T >
__inline__ __host__
T* HostPool::getElement( int index )
{
	ubyte* pElementStart = &( m_backingStore[ index * m_elementSizeBytes ] );
	return reinterpret_cast< T* >( pElementStart );
}
