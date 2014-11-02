#include "io/BinaryFileOutputStream.h"

// ==============================================================
// Public
// ==============================================================

// static
BinaryFileOutputStream* BinaryFileOutputStream::open( const char* filename )
{
	FILE* fp = fopen( filename, "wb" );
	if( fp != NULL )
	{
		return new BinaryFileOutputStream( fp );
	}
	else
	{
		return NULL;
	}
}

// virtual
BinaryFileOutputStream::~BinaryFileOutputStream()
{
	close();
}

void BinaryFileOutputStream::close()
{
	fclose( m_pFilePointer );
}

bool BinaryFileOutputStream::writeInt( int i )
{
	size_t itemsWritten = fwrite( &i, sizeof( int ), 1, m_pFilePointer );
	return( itemsWritten == 1 );
}

bool BinaryFileOutputStream::writeFloat( float f )
{
	size_t itemsWritten = fwrite( &f, sizeof( float ), 1, m_pFilePointer );
	return( itemsWritten == 1 );
}

bool BinaryFileOutputStream::writeFloatArray( float f[], int nCount )
{
	size_t itemsWritten = fwrite( f, sizeof( float ), nCount, m_pFilePointer );
	return( itemsWritten == nCount );
}

// ==============================================================
// Private
// ==============================================================

BinaryFileOutputStream::BinaryFileOutputStream( FILE* pFilePointer ) :
	m_pFilePointer( pFilePointer )
{

}