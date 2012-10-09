#pragma once

// STL
#include <numeric> // for std::iota

#ifdef __CUDACC__
// cuda::thrust
#include <thrust/remove.h>
#endif

// libcgt
#include <common/BasicTypes.h>
#include <math/Arithmetic.h>

// local
#include "DeviceQueue.h"

template< typename UsedListTag >
struct UsedListEntry
{
	int poolIndex;
	int isDeleted; // TODO: can optimize later by packing it all into the tag
	UsedListTag tag;
};

template< typename UsedListTag >
struct KernelPool
{
	__inline__ __host__
	KernelPool()
	{

	}

	__inline__ __host__
	KernelPool
	(
		int capacity,
		int elementSizeBytes,

		KernelQueue< int > freeList,
		KernelQueue< UsedListEntry< UsedListTag > > usedList,
		KernelVector< ubyte > backingStore
	) :

		m_capacity( capacity ),
		m_elementSizeBytes( elementSizeBytes ),
		m_freeList( freeList ),
		m_usedList( usedList ),
		m_backingStore( backingStore )

	{

	}

#ifdef __CUDACC__

	// gives you an entry that comes with an index
	// and space for a tag
	__inline__ __device__
	UsedListEntry< UsedListTag >& alloc()
	{
		int index = m_freeList.dequeue();

		UsedListEntry< UsedListTag > entry;
		entry.poolIndex = index;
		entry.isDeleted = 0;		

		uint usedListIndex = m_usedList.enqueue( entry );
		return m_usedList.ringBuffer()[ usedListIndex ];
	}
#endif
	
	// marks the index at usedList()[ usedListIndex ] as deleted
	__inline__ __device__
	void markAsDeleted( int usedListIndex )
	{
		UsedListEntry< UsedListTag >& entry = m_usedList.ringBuffer()[ usedListIndex ];
		entry.isDeleted = 1;
	}

	__inline__ __device__
	KernelQueue< int >& freeList()
	{
		return m_freeList;
	}

	__inline__ __device__
	KernelQueue< UsedListEntry< UsedListTag > >& usedList()
	{
		return m_usedList;
	}

	// given an index (returned from alloc()),
	// returns a pointer to the beginning of the element
	template< typename T >
	__inline__ __device__
	T* getElement( int index )
	{
		ubyte* pElementStart = &( m_backingStore[ index * m_elementSizeBytes ] );
		return reinterpret_cast< T* >( pElementStart );
	}

	int m_capacity;
	int m_elementSizeBytes;

	KernelQueue< int > m_freeList;
	KernelQueue< UsedListEntry< UsedListTag > > m_usedList;
	KernelVector< ubyte > m_backingStore;
};

template< typename UsedListTag >
class DevicePool
{
public:

	// creates a null pool
	DevicePool();

	// creates a device memory pool of capacity elements,
	// where each element occupies elementSizeBytes bytes
	//
	// capacity must be a power of two
	DevicePool( int capacity, int elementSizeBytes );
	virtual ~DevicePool();

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

	KernelPool< UsedListTag > kernelPool();	
	
//private:

	int m_capacity;
	int m_elementSizeBytes;

	// stores a list of free (still-available) element (not byte) indices in m_backingStore
	// initially free
	DeviceQueue< int > md_freeList;

	// stores a list of allocated (not currently available) element indices in m_backingStore
	// if the value is positive, then it's still in use
	// if the value is negative, then it's been marked as freed and will be collected in the next call to collect()
	DeviceQueue< UsedListEntry< UsedListTag > > md_usedList;

	// temporary storage for the garbage collector
	DeviceVector< UsedListEntry< UsedListTag > > md_collectedList;

	// backing store of capacity * elementSizeBytes bytes
	DeviceVector< ubyte > m_backingStore;
};

template< typename UsedListTag >
__inline__ __host__
DevicePool< UsedListTag >::DevicePool() :

	m_capacity( -1 ),
	m_elementSizeBytes( -1 )

{

}

template< typename UsedListTag >
__inline__ __host__
DevicePool< UsedListTag >::DevicePool( int capacity, int elementSizeBytes )
{
	resize( capacity, elementSizeBytes );
}

// virtual
template< typename UsedListTag >
__inline__ __host__
DevicePool< UsedListTag >::~DevicePool()
{

}

template< typename UsedListTag >
__inline__ __host__
bool DevicePool< UsedListTag >::isNull() const
{
	return( md_freeList.isNull() || m_backingStore.isNull() );
}

template< typename UsedListTag >
__inline__ __host__
bool DevicePool< UsedListTag >::notNull() const
{
	return !isNull();
}

template< typename UsedListTag >
__inline__ __host__
int DevicePool< UsedListTag >::capacity() const
{
	return m_capacity;
}

template< typename UsedListTag >
__inline__ __host__
int DevicePool< UsedListTag >::elementSizeBytes() const
{
	return m_elementSizeBytes;
};

template< typename UsedListTag >
__inline__ __host__
size_t DevicePool< UsedListTag >::sizeInBytes() const
{
	size_t esb = m_elementSizeBytes;
	size_t poolSizeBytes = esb * capacity();

	return poolSizeBytes + md_freeList.sizeInBytes() + md_usedList.sizeInBytes() + md_collectedList.sizeInBytes();
}


template< typename UsedListTag >
__inline__ __host__
int DevicePool< UsedListTag >::numFreeElements()
{
	return md_freeList.count();
}

template< typename UsedListTag >
__inline__ __host__
void DevicePool< UsedListTag >::resize( int capacity, int elementSizeBytes )
{
	m_capacity = capacity;
	m_elementSizeBytes = elementSizeBytes;
	md_freeList.resize( capacity );
	md_usedList.resize( capacity );
	md_collectedList.resize( capacity );
	m_backingStore.resize( capacity * elementSizeBytes );

	clear();	
}

template< typename UsedListTag >
__inline__ __host__
void DevicePool< UsedListTag >::clear()
{
	// TODO: thrust::generate?
	// generate free list: [0,capacity)
	std::vector< int > h_freeList( m_capacity );
	std::iota( h_freeList.begin(), h_freeList.end(), 0 );

	md_freeList.copyFromHost( h_freeList );
}

template< typename UsedListTag >
__inline__ __host__
KernelPool< UsedListTag > DevicePool< UsedListTag >::kernelPool()
{
	return KernelPool< UsedListTag >
	(
		m_capacity, m_elementSizeBytes,
		md_freeList.kernelQueue(),
		md_usedList.kernelQueue(),
		m_backingStore.kernelVector()
	);
}
