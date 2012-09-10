#pragma once

#include <cassert>

#include "DeviceVariable.h"
#include "DeviceVector.h"

#include "MathUtils.h"
#include "KernelQueue.h"

// An atomic producer-consumer queue for CUDA
// implemented using a circular buffer
// the buffer size *must be* a power of two
template< typename T >
class DeviceQueue
{
public:

	// initializes a null queue
	DeviceQueue();

	// initializes a queue with a fixed *power of two* length
	DeviceQueue( uint length );

	bool isNull() const;
	bool notNull() const;

	// resizes the queue:
	//   destroys the existing data
	//   and clears the queue in the process (head and tail set to 0)
	//
	// length *must be* a power of two (otherwise returns false)
	bool resize( uint length );

	// clears the queue
	void clear();

	// number of enqueued items
	int count();

	KernelQueue< T > kernelQueue();

	// copies count() elements from host --> device queue
	// this is automatically resized to src.size()
	// src.size() *must be* a power of two
	// and head must be first
	bool copyFromHost( const std::vector< T >& src );

	// copies count() elements from device queue --> host
	// dst is automatically resized and the head of the queue is first
	void copyToHost( std::vector< T >& dst ) const;

private:

	DeviceVariable< uint2 > m_headTail;
	DeviceVector< T > m_elements;

};

#include "DeviceQueue.inl"
