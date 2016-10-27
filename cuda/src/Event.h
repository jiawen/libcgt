#pragma once

#include <cuda_runtime.h>

// A simple C++ class wrapping the cudaEvent C API.
namespace libcgt { namespace cuda {

class Event
{
public:

    Event();
    ~Event();

    // records the start event
    void recordStart( cudaStream_t stream = 0 );

    // records the stop event
    void recordStop( cudaStream_t stream = 0 );

    // synchronizes the stream and returns the elapsed time
    float synchronizeAndGetMillisecondsElapsed();

    // same as stop() and then synchronizeAndGetMillisecondsElapsed()
    float recordStopSyncAndGetMillisecondsElapsed( cudaStream_t stream = 0 );

private:

    cudaEvent_t m_start;
    cudaEvent_t m_stop;

};

} } // cuda // libcgt
