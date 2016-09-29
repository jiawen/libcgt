#pragma once

#include <cstdint>
#include <string>
#include <vector>

class File
{
public:

    // Returns true if the filename is an existing file.
    // Does not work on directories.
    static bool exists( const char* filename );

    // Returns 0 on failure.
    static size_t size( const char* filename );

    // Reads a text file and returns it as an std::string.
    // Note that the string length may be less than fileSize
    // due to fread() converting "\r\n" to "\n".
    //
    // On failure, returns the empty string.
    static std::string readTextFile( const char* filename );

    // Reads an entire file into a vector of bytes, returning it.
    static std::vector< uint8_t > readBinaryFile( const char* filename );

    // Read a text file into a vector of strings, one per line.
    // The newline character at the end of each line is *removed*.
    static std::vector< std::string > readLines( const char* filename );
};
