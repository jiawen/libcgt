template< typename T >
__inline__ __host__
KernelQueue< T >::KernelQueue( uint2* d_pHeadTail, KernelVector< T > elements ) :
	md_pHeadTail( d_pHeadTail ),
	m_elements( elements )
{

}

#ifdef __CUDACC__

template< typename T >
__inline__ __device__
void KernelQueue< T >::enqueue( const T& val )
{
	uint oldTail = atomicAdd( tailPointer(), 1 );
	oldTail = libcgt::cuda::modPowerOfTwo( oldTail, m_elements.length );
	m_elements[ oldTail ] = val;
}	

template< typename T >
__inline__ __device__
T KernelQueue< T >::dequeue()
{
	uint oldHead = atomicAdd( headPointer(), 1 );
	oldHead = libcgt::cuda::modPowerOfTwo( oldHead, m_elements.length );
	return m_elements[ oldHead ];
}

#endif

template< typename T >
__inline__ __device__
int KernelQueue< T >::count()
{
	int head = *( headPointer() );
	int tail = *( tailPointer() );
	return( tail - head );
}

template< typename T >
__inline__ __device__
bool KernelQueue< T >::isFull()
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

template< typename T >
__inline__ __device__
bool KernelQueue< T >::isEmpty()
{
	int head = *( headPointer() );
	int tail = *( tailPointer() );
	return ( tail == head );
}

template< typename T >
__inline__ __device__
KernelVector< T >& KernelQueue< T >::elements()
{
	return m_elements;
}

template< typename T >
__inline__ __device__
uint* KernelQueue< T >::headPointer()
{
	return &( md_pHeadTail->x );
}

template< typename T >
__inline__ __device__
uint* KernelQueue< T >::tailPointer()
{
	return &( md_pHeadTail->y );
}

#if 0
	// tries to enqueue a value
	// returns false if the queue is full
	__inline__ __device__
		bool tryEnqueue( const T& val )
	{
		// use some magic trick with atomicInc with elements.length?	
	}

	// tries to dequeue a value
	// returns false if 
	__inline__ __device__
		bool tryDequeue( T& val )
	{
		uint tail = *( tailPointer() );
		// atomicInc computes:
		// uint oldHead = *headPointer;
		// if( oldHead >= tail )
		// {
		//     return 0;
		// }
		// else
		// {
		//     *headPointer = oldHead + 1
		//     return oldHead;
		// }
		uint retVal = atomicInc( headPointer(), tail );
		return ( retVal != 0 );
	}
#endif
