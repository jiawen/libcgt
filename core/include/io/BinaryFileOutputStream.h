#pragma once

#include <cstdio>

class BinaryFileOutputStream
{
public:

	static BinaryFileOutputStream* open( const char* filename );
	virtual ~BinaryFileOutputStream();

	void close();

	bool writeInt( int i );
	bool writeFloat( float f );
	bool writeFloatArray( float f[], int nCount );

private:

	BinaryFileOutputStream( FILE* pFilePointer );

	FILE* m_pFilePointer;
};
