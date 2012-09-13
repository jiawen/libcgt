#pragma once

#include <numeric>

#include <common/BasicTypes.h>
#include <math/Arithmetic.h>

#include "DeviceQueue.h"

// TODO: this can actually be compiled!

struct KernelPool
{
	__inline__ __host__
	KernelPool( int _capacity,
		int _elementSizeBytes,

		KernelQueue< int > _freeList,
		KernelVector< ubyte > _backingStore ) :

		capacity( _capacity ),
		elementSizeBytes( _elementSizeBytes ),
		freeList( _freeList ),
		backingStore( _backingStore )

	{

	}

#ifdef __CUDACC__

	// gives you an index
	__inline__ __device__
	int alloc()
	{
		int index = freeList.dequeue();
		return index;
	}
#endif

	// void dealloc(

	// given an index (returned from alloc()),
	// returns a pointer to the beginning of the element
	template< typename T >
	__inline__ __device__
	T* getElement( int index )
	{
		ubyte* pElementStart = &( backingStore[ index * elementSizeBytes ] );
		return reinterpret_cast< T* >( pElementStart );
	}

	int capacity;
	int elementSizeBytes;

	//KernelQueue< int > usedList;
	KernelQueue< int > freeList;
	KernelVector< ubyte > backingStore;
};

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

	// returns the number of free elements
	// (potentially expensive - incurs a GPU copy)
	int numFreeElements();

	// resizes the pool to a new capacity, clearing it at the same time
	void resize( int capacity, int elementSizeBytes );
	
	// clears the pool: marks all elements as free
	// does *not* touch the data
	void clear();

	KernelPool kernelPool();

	
//private:

	int m_capacity;
	int m_elementSizeBytes;
	//DeviceQueue< int > m_usedList;
	DeviceQueue< int > md_freeList;
	DeviceVector< ubyte > m_backingStore;
};

__inline__ __host__
DevicePool::DevicePool() :

	m_capacity( -1 ),
	m_elementSizeBytes( -1 )

	//m_usedList( count ),
{

}

__inline__ __host__
DevicePool::DevicePool( int capacity, int elementSizeBytes )
	//m_usedList( count ),
{
	resize( capacity, elementSizeBytes );
}

// virtual
__inline__ __host__
DevicePool::~DevicePool()
{

}

__inline__ __host__
bool DevicePool::isNull() const
{
	return( md_freeList.isNull() || m_backingStore.isNull() );
}

__inline__ __host__
bool DevicePool::notNull() const
{
	return !isNull();
}

__inline__ __host__
int DevicePool::capacity() const
{
	return m_capacity;
}

__inline__ __host__
int DevicePool::numFreeElements()
{
	return md_freeList.count();
}

__inline__ __host__
void DevicePool::resize( int capacity, int elementSizeBytes )
{
	m_capacity = capacity;
	m_elementSizeBytes = elementSizeBytes;
	md_freeList.resize( capacity );
	m_backingStore.resize( capacity * elementSizeBytes );

	clear();	
}

__inline__ __host__
void DevicePool::clear()
{
	// generate free list: [0,capacity)
	std::vector< int > h_freeList( m_capacity );
	std::iota( h_freeList.begin(), h_freeList.end(), 0 );

	md_freeList.copyFromHost( h_freeList );
}

__inline__ __host__
KernelPool DevicePool::kernelPool()
{
	return KernelPool
	(
		m_capacity, m_elementSizeBytes,
		md_freeList.kernelQueue(),
		m_backingStore.kernelVector()
	);
}
