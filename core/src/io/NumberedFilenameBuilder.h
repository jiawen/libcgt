#pragma once

#include <QString>

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
    NumberedFilenameBuilder( QString prefix, QString suffix, int nDigits = 5 );

    QString filenameForNumber( int number );

private:

    int m_nDigits;
    QString m_baseString;

};