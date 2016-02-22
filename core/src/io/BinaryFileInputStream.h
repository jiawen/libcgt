#pragma once

#include <cstdio>

class BinaryFileInputStream
{
public:

    static BinaryFileInputStream* open( const char* filename );
    virtual ~BinaryFileInputStream();

    void close();

    bool readInt( int* i );
    bool readIntArray( int[], int nCount );

    bool readFloat( float* f );
    bool readFloatArray( float f[], int nCount );

private:

    BinaryFileInputStream( FILE* pFilePointer );

    FILE* m_pFilePointer;
};
