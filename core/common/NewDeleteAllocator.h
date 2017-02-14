#pragma once

#include "libcgt/core/common/Allocator.h"

// An Allocator that uses C++'s new[] and delete[].
class NewDeleteAllocator : public Allocator
{
public:

    // Singleton instance.
    static NewDeleteAllocator* instance();

    virtual void* allocate( size_t bytes );
    virtual void deallocate( void* pointer, size_t bytes );

private:

    static NewDeleteAllocator s_singleton;
};
