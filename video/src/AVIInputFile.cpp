#include "AVIInputFile.h"

// static
AVIInputFile* AVIInputFile::open( QString filename )
{
	AVIFileInit();

	PAVIFILE pAVIFile;
	HRESULT hr = AVIFileOpen
	(
		&pAVIFile,
		reinterpret_cast< LPCWSTR >( filename.utf16() ), // wcharFilename.data(),
		OF_READ,
		NULL // default handler
	);

	if( FAILED( hr ) )
	{
		// AVIERR_BADFORMAT / REGDB_E_CLASSNOTREG: unrecgonized format
		// AVIERR_MEMORY: out of memory
		// AVIERR_FILEREAD / AVIERR_FILEOPEN error opening file

		AVIFileExit();
		return nullptr;
	}
	
	AVIFILEINFO fileInfo;
	hr = AVIFileInfo( pAVIFile, &fileInfo, sizeof( AVIFILEINFO ) );

	if( FAILED( hr ) )
	{
		AVIFileClose( pAVIFile );
		AVIFileExit();
		return nullptr;
	}

	AVIInputFile* pInputStream = new AVIInputFile;
	pInputStream->m_pAVIFile = pAVIFile;
	pInputStream->m_fileInfo = fileInfo;
	return pInputStream;	
}

// virtual
AVIInputFile::~AVIInputFile()
{
	AVIFileRelease( m_pAVIFile );
	AVIFileExit();
}

int AVIInputFile::numStreams() const
{
	return m_fileInfo.dwStreams;
}

int AVIInputFile::width() const
{
	return m_fileInfo.dwWidth;
}

int AVIInputFile::height() const
{
	return m_fileInfo.dwHeight;
}

Vector2i AVIInputFile::size() const
{
    return{ width(), height() };
}

int AVIInputFile::numFrames() const
{
	return m_fileInfo.dwLength;
}

float AVIInputFile::framesPerSecond() const
{
	return Arithmetic::divideIntsToFloat( m_fileInfo.dwRate, m_fileInfo.dwScale );
}

void AVIInputFile::framesPerSecondRational( int& numerator, int& denominator ) const
{
	numerator = m_fileInfo.dwRate;
	denominator = m_fileInfo.dwScale;
}

bool AVIInputFile::allKeyFrames() const
{	
	return m_fileInfo.dwCaps & AVIFILECAPS_ALLKEYFRAMES;
}

bool AVIInputFile::uncompressed() const
{
	return m_fileInfo.dwCaps & AVIFILECAPS_NOCOMPRESSION;
}

AVIInputFile::AVIInputFile()
{	
}

AVIInputVideoStream* AVIInputFile::openVideoStream( int streamIndex )
{
	return AVIInputVideoStream::open( m_pAVIFile, streamIndex, width(), height() );
}
