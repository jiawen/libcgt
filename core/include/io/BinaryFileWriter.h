#pragma once

#include <cstdio>

// TODO: rename to binary file output stream
class BinaryFileWriter
{
public:

	static BinaryFileWriter* open( const char* filename );
	virtual ~BinaryFileWriter();

	void close();

	bool writeInt( int i );
	bool writeFloat( float f );
	bool writeFloatArray( float f[], int nCount );

private:

	BinaryFileWriter( FILE* pFilePointer );

	FILE* m_pFilePointer;
};
