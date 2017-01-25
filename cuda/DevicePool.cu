#include "libcgt/cuda/DevicePool.h"

// thrust
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

// libcgt
#include "libcgt/core/common/ArrayUtils.h"
#include "libcgt/core/common/ArrayView.h"

using libcgt::core::arrayutils::writeViewOf;

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
    int* pDevicePointer = md_freeList.ringBuffer().pointer();
    thrust::device_ptr< int > pBegin = thrust::device_pointer_cast( pDevicePointer );
    thrust::device_ptr< int > pEnd = thrust::device_pointer_cast( pDevicePointer + m_capacity );

    thrust::sequence( pBegin, pEnd, 0 );

    md_freeList.setHeadAndTailAbsoluteIndices( 0, m_capacity );
}

std::vector< uint8_t > DevicePool::getElement( int index ) const
{
    // Allocate memory for the output.
    std::vector< uint8_t > output( elementSizeBytes() );

    // Copy it to the host.
    copy( md_backingStore, index * elementSizeBytes(), writeViewOf( output ) );

    return output;
}

KernelPool DevicePool::kernelPool()
{
    return KernelPool
    (
        m_capacity,
        m_elementSizeBytes,
        md_freeList.kernelQueue(),
        md_backingStore.writeView()
    );
}
