#include "libcgt/core/common/NewDeleteAllocator.h"

#include <cstdint>

// static
NewDeleteAllocator* NewDeleteAllocator::instance()
{
    return &s_singleton;
}

// virtual
void* NewDeleteAllocator::allocate( size_t bytes )
{
    return reinterpret_cast< void* >( new uint8_t[ bytes ] );
}

// virtual
void NewDeleteAllocator::deallocate( void* pointer, size_t bytes )
{
    uint8_t* byteArray = reinterpret_cast< uint8_t* >( pointer );
    delete[] byteArray;
}

NewDeleteAllocator NewDeleteAllocator::s_singleton;
