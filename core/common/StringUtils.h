#pragma once

#include <memory>
#include <iostream>
#include <string>
#include <cstdio>

namespace libcgt { namespace core {

// "printf" to an std::string. This function uses std::snprintf to determine
// the required buffer size, then dynamically allocate memory for it. As such,
// it will not perform well in inner loops.
template< typename ... Args >
std::string stringPrintf( const std::string& format, Args ... args )
{
    // Figure out how big of a buffer we need. The extra space is for '\0'.
    size_t size = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1;
    // Allocate memory for it. It will be automatically deleted.
    auto buffer = std::make_unique< char[] >( size );
    // Actually printf to it.
    snprintf( buffer.get(), size, format.c_str(), args ... );
    // We don't want the '\0' inside.
    return std::string( buffer.get(), buffer.get() + size - 1 );
}

} } // core, libcgt
