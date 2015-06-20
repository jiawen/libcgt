#include "Event.h"

libcgt::cuda::Event::Event()
{
    cudaEventCreate( &m_start );
    cudaEventCreate( &m_stop );
}

// virtual
libcgt::cuda::Event::~Event()
{
    cudaEventDestroy( m_stop );
    cudaEventDestroy( m_start );
}

void libcgt::cuda::Event::recordStart( cudaStream_t stream )
{
    cudaEventRecord( m_start, stream );
}

void libcgt::cuda::Event::recordStop( cudaStream_t stream )
{
    cudaEventRecord( m_stop, stream );
}

float libcgt::cuda::Event::synchronizeAndGetMillisecondsElapsed()
{
    cudaEventSynchronize( m_stop );

    float ms;
    cudaEventElapsedTime( &ms, m_start, m_stop );
    return ms;
}

float libcgt::cuda::Event::recordStopSyncAndGetMillisecondsElapsed( cudaStream_t stream )
{
    recordStop( stream );
    return synchronizeAndGetMillisecondsElapsed();
}