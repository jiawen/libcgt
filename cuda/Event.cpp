#include "libcgt/cuda/Event.h"

namespace libcgt { namespace cuda {

Event::Event()
{
    cudaEventCreate( &m_start );
    cudaEventCreate( &m_stop );
}

Event::~Event()
{
    cudaEventDestroy( m_stop );
    cudaEventDestroy( m_start );
}

void Event::recordStart( cudaStream_t stream )
{
    cudaEventRecord( m_start, stream );
}

void Event::recordStop( cudaStream_t stream )
{
    cudaEventRecord( m_stop, stream );
}

float Event::synchronizeAndGetMillisecondsElapsed()
{
    cudaEventSynchronize( m_stop );

    float ms;
    cudaEventElapsedTime( &ms, m_start, m_stop );
    return ms;
}

float Event::recordStopSyncAndGetMillisecondsElapsed( cudaStream_t stream )
{
    recordStop( stream );
    return synchronizeAndGetMillisecondsElapsed();
}

} } // cuda // libcgt
