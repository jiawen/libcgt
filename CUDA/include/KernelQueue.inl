template< typename T >
__inline__ __host__
KernelQueue< T >::KernelQueue( uint2* d_pReadIndexAndCount, KernelVector< T > elements ) :

	md_pReadIndexAndCount( d_pReadIndexAndCount ),
	m_elements( elements )
{

}

#ifdef __CUDACC__

template< typename T >
__inline__ __device__
void KernelQueue< T >::enqueue( const T& val )
{
	// assuming not full
	//int writeIndex = ( md_pReadIndexAndCount->x + md_pReadIndexAndCount->y ) % capacity();
	//m_elements[ writeIndex ] = val;
	//++( md_pReadIndexAndCount->y );

	uint oldCount = atomicAdd( countPointer(), 1 );
	m_elements[ ( *( readIndexPointer() ) + oldCount ) % capacity() ] = val;
}

template< typename T >
__inline__ __device__
T KernelQueue< T >::dequeue()
{
	T output;

#if 1
	uint oldReadIndex = atomicAdd( readIndexPointer(), 1 );
	atomicSub( countPointer(), 1 );

	output = m_elements[ oldReadIndex % capacity() ];
#endif

	return output;
}

#endif

template< typename T >
__inline__ __device__
int KernelQueue< T >::count()
{
	return *( countPointer() );
}

template< typename T >
__inline__ __device__
int KernelQueue< T >::capacity() const
{
	return m_elements.length;
}

template< typename T >
__inline__ __device__
bool KernelQueue< T >::isFull()
{
	return( count() == capacity() );
}

template< typename T >
__inline__ __device__
bool KernelQueue< T >::isEmpty()
{
	return( count() == 0 );
}

template< typename T >
__inline__ __device__
KernelVector< T >& KernelQueue< T >::elements()
{
	return m_elements;
}

template< typename T >
__inline__ __device__
uint* KernelQueue< T >::readIndexPointer()
{
	return &( md_pReadIndexAndCount->x );
}

template< typename T >
__inline__ __device__
uint* KernelQueue< T >::countPointer()
{
	return &( md_pReadIndexAndCount->y );
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
