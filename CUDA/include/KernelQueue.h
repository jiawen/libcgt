
template< typename T >
struct KernelQueue
{
	__inline__ __host__
	KernelQueue( uint2* d_pHeadTail, KernelVector< T > elements );

#ifdef __CUDACC__

	// enqueues a value at the tail the queue
	// (atomically increment the tail pointer
	// and set the value where it used to be)
	__inline__ __device__
	void enqueue( const T& val );

	// removes an element from the head of the queue
	// (atomically increment the head pointer
	// and return the value where it used to be)
	__inline__ __device__
	T dequeue();

#endif

	__inline__ __device__
	int count();

	__inline__ __device__
	bool isFull();

	__inline__ __device__
	bool isEmpty();

	__inline__ __device__
	KernelVector< T >& elements();

private:

	__inline__ __device__
	uint* headPointer();

	__inline__ __device__
	uint* tailPointer();

	uint2* md_pHeadTail;
	KernelVector< T > m_elements;

};

#include "KernelQueue.inl"
