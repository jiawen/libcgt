#pragma once

#include <cstdio>
#include <vector>

#include <common/ArrayUtils.h>

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
	__inline__ __host__
	T* getElement( int index );

	void copyFromDevice( const DevicePool< UsedListTag >& pool );
	void copyToDevice( DevicePool< UsedListTag >& pool );

	void loadBinary( FILE* fp );
	void saveBinary( FILE* fp );

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

#include "HostPool.inl"
