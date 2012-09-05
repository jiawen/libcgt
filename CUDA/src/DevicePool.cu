#include "DevicePool.h"

DevicePool::DevicePool()
{

}

DevicePool::DevicePool( int capacity, int elementSizeBytes ) :

	m_capacity( capacity ),
	m_elementSizeBytes( elementSizeBytes ),

	//m_usedList( count ),
	m_freeList( capacity ),

	m_pool( capacity * elementSizeBytes )

{
	reset();
}

KernelPool DevicePool::kernelPool()
{
	return KernelPool( m_freeList.kernelQueue(), m_pool.kernelVector() );
}

void DevicePool::reset()
{
	// m_usedList.clear();

	// generate free list: [0,capacity)
	std::vector< int > h_freeList( m_capacity );
	std::iota( h_freeList.begin(), h_freeList.end(), 0 );

	m_freeList.elements().copyFromHost( h_freeList );
}
