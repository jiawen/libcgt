#include "DevicePool.h"

// thrust
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

// libcgt
#include <common/Array1DView.h>

DevicePool::DevicePool() :

	m_capacity( -1 ),
	m_elementSizeBytes( -1 )

{

}

DevicePool::DevicePool( int capacity, int elementSizeBytes )
{
	resize( capacity, elementSizeBytes );
}

// virtual
DevicePool::~DevicePool()
{

}

bool DevicePool::isNull() const
{
	return( md_freeList.isNull() || md_backingStore.isNull() );
}

bool DevicePool::notNull() const
{
	return !isNull();
}

int DevicePool::capacity() const
{
	return m_capacity;
}

int DevicePool::elementSizeBytes() const
{
	return m_elementSizeBytes;
};

size_t DevicePool::sizeInBytes() const
{
	size_t esb = m_elementSizeBytes;
	size_t poolSizeBytes = esb * capacity();

	return poolSizeBytes + md_freeList.sizeInBytes();
}

int DevicePool::numFreeElements()
{
	return md_freeList.count();
}

void DevicePool::resize( int capacity, int elementSizeBytes )
{
	m_capacity = capacity;
	m_elementSizeBytes = elementSizeBytes;
	md_freeList.resize( capacity );
	md_backingStore.resize( capacity * elementSizeBytes );

	clear();	
}

void DevicePool::clear()
{
	int* pDevicePointer = md_freeList.ringBuffer().devicePointer();
	thrust::device_ptr< int > pBegin = thrust::device_pointer_cast( pDevicePointer );
	thrust::device_ptr< int > pEnd = thrust::device_pointer_cast( pDevicePointer + m_capacity );

	thrust::sequence( pBegin, pEnd, 0 );

	md_freeList.setHeadAndTailAbsoluteIndices( 0, m_capacity );
}

std::vector< ubyte > DevicePool::getElement( int index ) const
{
	// allocate memory for the output
	std::vector< ubyte > output( elementSizeBytes() );

	// view it as a byte array
	Array1DView< ubyte > view( output.data(), elementSizeBytes() );

	// copy it to the host
	md_backingStore.copyToHost( view, index * elementSizeBytes() );

	return output;
}

KernelPool DevicePool::kernelPool()
{
	return KernelPool
	(
		m_capacity,
		m_elementSizeBytes,
		md_freeList.kernelQueue(),
		md_backingStore.kernelVector()
	);
}
