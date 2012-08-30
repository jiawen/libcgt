#pragma once

#include <numeric>

#include <common/BasicTypes.h>

#include "DeviceQueue.h"

// TODO: this can actually be compiled!

struct KernelPool
{
	__inline__ __host__
	KernelPool( KernelQueue< int > _freeList,
		KernelVector< ubyte > _pool ) :

		freeList( _freeList ),
		pool( _pool )

	{

	}

#ifdef __CUDACC__

	// gives you an index and a pointer
	__inline__ __device__
	int alloc()
	{
		int index = freeList.dequeue();
		return index;
	}
#endif

	// void dealloc(

	//KernelQueue< int > usedList;
	KernelQueue< int > freeList;
	KernelVector< ubyte > pool;
};

class DevicePool
{
public:

	DevicePool();

	DevicePool( int capacity, int elementSizeBytes );
	
	KernelPool kernelPool();
	
	void reset();

//private:

	int m_capacity;
	int m_elementSizeBytes;
	//DeviceQueue< int > m_usedList;
	DeviceQueue< int > m_freeList;
	DeviceVector< ubyte > m_pool;
};

__inline__ __host__
DevicePool::DevicePool()
{

}

__inline__ __host__
DevicePool::DevicePool( int capacity, int elementSizeBytes ) :

	m_capacity( capacity ),
	m_elementSizeBytes( elementSizeBytes ),

	//m_usedList( count ),
	m_freeList( capacity ),

	m_pool( capacity * elementSizeBytes )

{
	reset();
}

__inline__ __host__
KernelPool DevicePool::kernelPool()
{
	return KernelPool( m_freeList.kernelQueue(), m_pool.kernelVector() );
}

__inline__ __host__
void DevicePool::reset()
{
	// m_usedList.clear();

	// generate free list: [0,capacity)
	std::vector< int > h_freeList( m_capacity );
	std::iota( h_freeList.begin(), h_freeList.end(), 0 );

	m_freeList.elements().copyFromHost( h_freeList );
}
