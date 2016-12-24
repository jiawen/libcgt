#pragma once

#include <string>

class NumberedFilenameBuilder
{
public:

    // Construct a NumberedFilenameBuilder for a sequence of files
    //
    // e.g.,
    //   prefix = "/tmp/frame_"
    //   suffix = ".png"
    //   output = "/tmp/frame_00000.png"
    //
    // prefix cannot contain the string "%1" as a substring
    // (a bad idea for filenames anyway...)
    NumberedFilenameBuilder( const std::string& prefix,
                            const std::string& suffix,
                            int nDigits = 5 );

    std::string filenameForNumber( int number );

private:

    std::string m_prefix;
    std::string m_suffix;
    int m_nDigits;

};