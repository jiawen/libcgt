#include <ThreadMath.cuh>

template< typename T >
struct KernelQueue
{
	__inline__ __host__
	KernelQueue();

	__inline__ __host__
	KernelQueue( uint2* d_pReadIndexAndCount, KernelVector< T > elements );

#ifdef __CUDACC__
	// enqueues a value at the tail the queue
	// (atomically increment the tail pointer
	// and set the value where it used to be)
	// returns the index in ringBuffer() where the element was placed
	__inline__ __device__
	uint enqueue( const T& val );

	// removes an element from the head of the queue
	// (atomically increment the head pointer
	// and return the value where it used to be)
	__inline__ __device__
	T dequeue();
#endif

	__inline__ __device__
	int capacity() const;

	__inline__ __device__
	int count();

	__inline__ __device__
	bool isEmpty();

	__inline__ __device__
	bool isFull();

	__inline__ __device__
	KernelVector< T >& ringBuffer();

	// pointer to the head index
	// &( md_pHeadTailAbsoluteIndices->x )
	__inline__ __device__
	uint* headIndexPointer();

	// pointer to the tail index
	// &( md_pHeadTailAbsoluteIndices->y )
	__inline__ __device__
	uint* tailIndexPointer();

private:	

	uint2* md_pHeadTailAbsoluteIndices;
	KernelVector< T > md_ringBuffer;

};

#include "KernelQueue.inl"
