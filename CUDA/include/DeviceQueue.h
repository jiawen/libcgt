#pragma once

#include <cassert>

#include "DeviceVariable.h"
#include "DeviceVector.h"

#include "MathUtils.h"

// An atomic producer-consumer queue for CUDA
// implemented using a circular buffer
// the buffer size *must be* a power of two

template< typename T >
struct KernelQueue
{
#if 0
	__inline__ __host__
	KernelQueue( uint2* d_pHeadTail, KernelVector< T > elements ) :
		md_pHeadTail( d_pHeadTail ),
		m_elements( elements )
	{

	}
#endif

	__inline__ __host__
	KernelQueue( uint* d_pHead, uint* d_pTail, KernelVector< T > elements ) :
		md_pHead( d_pHead ),
		md_pTail( d_pTail ),
		m_elements( elements )
	{

	}

#ifdef __CUDACC__

	// enqueues a value at the tail the queue
	// (atomically increment the tail pointer
	// and set the value where it used to be)
	__inline__ __device__
	void enqueue( const T& val )
	{
		//printf( "enqueue: tailPointer = 0x%p\n", tailPointer() );
		uint oldTail = atomicAdd( tailPointer(), 1 );
		oldTail = libcgt::cuda::modPowerOfTwo( oldTail, m_elements.length );
		m_elements[ oldTail ] = val;
	}	

	// removes an element from the head of the queue
	// (atomically increment the head pointer
	// and return the value where it used to be)
	__inline__ __device__
	T dequeue()
	{
		uint oldHead = atomicAdd( headPointer(), 1 );
		oldHead = libcgt::cuda::modPowerOfTwo( oldHead, m_elements.length );
		return m_elements[ oldHead ];
	}

#if 0
	// tries to enqueue a value
	// returns false if the queue is full
	__inline__ __device__
	bool tryEnqueue( const T& val )
	{
		// use some magic trick with atomicInc with elements.length?	
	}
#endif

	__inline__ __device__
	bool isFull()
	{
		// we can distinguish between full and empty
		// because we use absolute indices:
		//
		// if we ever wrote anything to the queue
		// then if they point to the same place after wraparound
		// then we're full
		int tail = *( tailPointer() );
		if( tail > 0 )
		{
			int head = *( headPointer() );
			return( ( tail - head ) < m_elements.length );
		}
		// if we never wrote anything to the queue, then it clearly can't be full
		else
		{
			return false;
		}
	}

	__inline__ __device__
	bool isEmpty()
	{
		int head = *( headPointer() );
		int tail = *( tailPointer() );
		return ( tail == head );
	}

	__inline__ __device__
	KernelVector< T >& elements()
	{
		return m_elements;
	}

	__inline__ __device__
	uint* headPointer()
	{
		return md_pHead;
		//return &( md_pHeadTail->x );
	}

	__inline__ __device__
	uint* tailPointer()
	{
		return md_pTail;
		//return &( md_pHeadTail->y );
	}
#endif

private:

	uint* md_pHead;
	uint* md_pTail;
	KernelVector< T > m_elements;

};

template< typename T >
class DeviceQueue
{
public:

	DeviceQueue();
	DeviceQueue( uint length );

	bool isNull() const;
	bool notNull() const;

	// resizes the queue (destroys the existing data and clear it in the process)
	void resize( uint length );

	// clears the queue
	void clear();

	// number of enqueued items
	int count();

	KernelQueue< T > kernelQueue();

	// for debugging
	DeviceVector< T >& elements();

private:

	// DeviceVariable< uint2 > m_headTail;
	DeviceVariable< uint > m_head;
	DeviceVariable< uint > m_tail;
	DeviceVector< T > m_elements;

};

template< typename T >
DeviceQueue< T >::DeviceQueue()
{

}

template< typename T >
DeviceQueue< T >::DeviceQueue( uint length ) :

	// m_headTail( make_uint2( 0, 0 ) ),
	m_head( 0 ),
	m_tail( 0 ),
	m_elements( length )
{
	assert( libcgt::cuda::isPowerOfTwo( length ) );
}

template< typename T >
bool DeviceQueue< T >::isNull() const
{
	return m_elements.isNull();
}

template< typename T >
bool DeviceQueue< T >::notNull() const
{
	return !isNull();
}

template< typename T >
void DeviceQueue< T >::resize( uint length )
{
	m_elements.resize( length );
	clear();
}

template< typename T >
void DeviceQueue< T >::clear()
{
	// m_headTail.set( make_uint2( 0, 0 ) );
	m_head.set( 0u );
	m_tail.set( 0u );
}

template< typename T >
int DeviceQueue< T >::count()
{
	// uint2 ht = m_headTail.get();
	// return( ht.y - ht.x );

	uint h = m_head.get();
	uint t = m_tail.get();
	return( t - h );
}

template< typename T >
KernelQueue< T > DeviceQueue< T >::kernelQueue()
{
	return KernelQueue< T >( m_head.devicePointer(), m_tail.devicePointer(), m_elements.kernelVector() );
	//return KernelQueue< T >( m_headTail.devicePointer(), m_elements.kernelVector() );
}

template< typename T >
DeviceVector< T >& DeviceQueue< T >::elements()
{
	return m_elements;
}
