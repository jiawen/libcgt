#pragma once

#include "DeviceVector.h"

// A simple queue for CUDA
// It is *not* a circular buffer
// and only supports enqueueing and dequeueing once
// without proper wraparound
template< typename T >
struct KernelQueue
{
	__inline__ __host__
	KernelQueue( uint* _pHead, uint* _pTail, KernelVector< T > _elements ) :
		pHead( _pHead ),
		pTail( _pTail ),
		elements( _elements )
	{

	}

	// add an element at the tail the queue
	// (atomically increment the tail pointer
	// and set the value where it used to be)

	__inline__ __device__
	void enqueue( const T& element )
	{
		uint oldTail = atomicInc( pTail, 0 );
		elements[ oldTail ] = element;
	}

#if 0
	bool tryEnqueue( const T& element )
	{
		// atomicInc tests:
		// if( *pTail >= m_elements.length )
		//   return 0
		// else
		//   return( *pTail + 1 );
		uint oldTail = atomicInc( pTail, m_elements.length );
		//elements[ oldTail % m_elements.length ] = element;
	}
#endif

	// removes an element from the head of the queue
	// (atomically increment the head pointer
	// and return the value where it used to be)
	__inline__ __device__
	T dequeue()
	{
		uint oldHead = atomicInc( pHead, 0 );
		elements[ oldHead ] = element;
	}

	uint* pHead;
	uint* pTail;
	KernelVector< T > elements;

};

template< typename T >
class DeviceQueue
{
public:

	DeviceQueue( uint length ) :
		m_elements( length )
	{
		cudaMalloc< uint >( &md_pHeadTail, 2 * sizeof( uint ) );
		md_pHead = &( md_pHeadTail[0] );
		md_pTail = &( md_pHeadTail[1] );

		reset();
	}

	~DeviceQueue()
	{
		cudaFree( md_pHeadTail );
		md_pHead = nullptr;
		md_pTail = nullptr;
	}

	// TODO: write kernels for clearing things when they're not 0
	void reset()
	{
		cudaMemset( md_pHeadTail, 0, 2 * sizeof( uint ) );
	}

	// numbers of enqueued items
	int count()
	{
		uint headTail[2];
		cudaMemcpy( headTail, md_pHeadTail, 2 * sizeof( uint ), cudaMemcpyDeviceToHost );
		return headTail[1] - headTail[0];
	}

	KernelQueue< T > kernelQueue()
	{
		return KernelQueue< T >( md_pHead, md_pTail, m_elements.kernelVector() );
	}

	DeviceVector< T >& elements()
	{
		return m_elements;
	}

private:

	uint* md_pHeadTail;
		uint* md_pHead; // &( md_pHeadTail[0] )
		uint* md_pTail; // &( md_pHeadTail[1] )

	DeviceVector< T > m_elements;

};