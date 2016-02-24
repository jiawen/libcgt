#pragma once

// An interface for a memory allocator.
class Allocator
{
public:

    virtual void* allocate( size_t bytes ) = 0;
    virtual void deallocate( void* pointer, size_t bytes ) = 0;
};