#pragma once

// STL
#include <vector>

// thrust
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

// libcgt
#include <common/Array1DView.h>
#include <common/BasicTypes.h>
#include <math/Arithmetic.h>

// local
#include "DeviceQueue.h"

struct KernelPool
{
	__inline__ __host__ __device__
	KernelPool();

	__inline__ __host__ __device__
	KernelPool
	(
		int capacity,
		int elementSizeBytes,

		KernelQueue< int > freeList,		
		KernelVector< ubyte > backingStore
	);

	// allocates an element and gives you back an index
	__inline__ __device__
	int alloc();

	// returns an index back to the pool
	__inline__ __device__
	void free( int index );	

	__device__
	KernelQueue< int >& freeList();

	// given an index (returned from alloc()),
	// returns a pointer to the beginning of the element
	template< typename T >
	__device__
	T* getElement( int index );

	int m_capacity;
	int m_elementSizeBytes;

	KernelQueue< int > m_freeList;
	KernelVector< ubyte > m_backingStore;
};

__inline__ __host__ __device__
KernelPool::KernelPool()
{

}

__inline__ __host__ __device__
KernelPool::KernelPool
(
	int capacity,
	int elementSizeBytes,

	KernelQueue< int > freeList,
	KernelVector< ubyte > backingStore
) :

	m_capacity( capacity ),
	m_elementSizeBytes( elementSizeBytes ),
	m_freeList( freeList ),
	m_backingStore( backingStore )

{

}

#ifdef __CUDACC__
__inline__ __device__
int KernelPool::alloc()
{
	int index = m_freeList.dequeue();
	return index;
}

__inline__ __device__
void KernelPool::free( int index )
{
	m_freeList.enqueue( index );
}
#endif

__inline__ __device__
KernelQueue< int >& KernelPool::freeList()
{
	return m_freeList;
}

template< typename T >
__inline__ __device__
T* KernelPool::getElement( int index )
{
	ubyte* pElementStart = &( m_backingStore[ index * m_elementSizeBytes ] );
	return reinterpret_cast< T* >( pElementStart );
}

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

	// get an element from the device
	// WARNING: probably slow as it incurs a cudaMemcpy	
	std::vector< ubyte > getElement( int index ) const;

	KernelPool kernelPool();	
	
//private:

	int m_capacity;
	int m_elementSizeBytes;

	// stores a list of free (still-available) element (not byte) indices in md_backingStore
	// initially, every element is free
	// and so is an entirely full queue of [0 1 2 3 ... m_capacity)
	// head = 0, tail = m_capacity
	DeviceQueue< int > md_freeList;	

	// backing store of capacity * elementSizeBytes bytes
	DeviceVector< ubyte > md_backingStore;
};
