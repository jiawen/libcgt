#include "MediaFoundationOutputVideoStream.h"

#include <QString>

#include <common/ArrayUtils.h>
#include <imageproc/Swizzle.h>
#include <math/Arithmetic.h>

#pragma comment( lib, "mfreadwrite" )
#pragma comment( lib, "mfplat" )
#pragma comment( lib, "mfuuid" )

// static
int MediaFoundationOutputVideoStream::recommendedBitsPerSecond( int width, int height, int framesPerSecondNumerator,
	int framesPerSecondDenominator )
{
	// preserve this ratio
	// 6 megabits per second for 1080p24
	const float fullHDSamplesPerSecond = 1920 * 1080 * 24;
	const float fullHDBitsPerSecond = 6 * 1024 * 1024;

	float inputSamplesPerSecond = width * height * framesPerSecondNumerator / framesPerSecondDenominator;
	return Arithmetic::roundToInt( inputSamplesPerSecond * fullHDBitsPerSecond / fullHDSamplesPerSecond );
}

// static
MediaFoundationOutputVideoStream* MediaFoundationOutputVideoStream::open( QString filename,
	Codec codec,
	int width, int height,
	int framesPerSecondNumerator, int framesPerSecondDenominator,
	int bitsPerSecond )
{
	MediaFoundationOutputVideoStream* pStream = nullptr;

	HRESULT hr = CoInitializeEx( nullptr, COINIT_APARTMENTTHREADED );
	if( SUCCEEDED( hr ) )
	{
		hr = MFStartup( MF_VERSION );

		if( SUCCEEDED( hr ) )
		{
			pStream = new MediaFoundationOutputVideoStream
			(
				filename,
				codec,
				width, height,
				framesPerSecondNumerator, framesPerSecondDenominator,
				bitsPerSecond
			);

			if( !( pStream->m_valid ) )
			{
				delete pStream;
				pStream = nullptr;
			}
		}
	}
	
	return pStream;
}

// virtual
MediaFoundationOutputVideoStream::~MediaFoundationOutputVideoStream()
{
	if( m_pBuffer != nullptr )
	{
		m_pBuffer->Release();
		m_pBuffer = nullptr;
	}

	if( m_pSample != nullptr )
	{
		m_pSample->Release();
		m_pSample = nullptr;
	}

	if( m_pMediaTypeIn != nullptr )
	{
		m_pMediaTypeIn->Release();
		m_pMediaTypeIn = nullptr;
	}

	if( m_pMediaTypeOut != nullptr )
	{
		m_pMediaTypeOut->Release();
		m_pMediaTypeOut = nullptr;
	}

	if( m_pSinkWriter != nullptr )
	{
		m_pSinkWriter->Release();
		m_pSinkWriter = nullptr;
	}

	MFShutdown();
	CoUninitialize();
}

MediaFoundationOutputVideoStream::Codec MediaFoundationOutputVideoStream::codec() const
{
	return m_codec;
}

int MediaFoundationOutputVideoStream::width() const
{
	return m_width;
}

int MediaFoundationOutputVideoStream::height() const
{
	return m_height;
}

Vector2i MediaFoundationOutputVideoStream::size() const
{
	return Vector2i( m_width, m_height );
}

float MediaFoundationOutputVideoStream::framesPerSecond() const
{
	return Arithmetic::divideIntsToFloat( m_framesPerSecondNumerator, m_framesPerSecondDenominator );
}

Vector2i MediaFoundationOutputVideoStream::framesPerSecondRational() const
{
	return Vector2i( m_framesPerSecondNumerator, m_framesPerSecondDenominator );
}

int64 MediaFoundationOutputVideoStream::frameDuration() const
{
	return static_cast< int64 >( m_sampleDuration );
}

float MediaFoundationOutputVideoStream::frameDurationMilliseconds() const
{
	return 1000.f * Arithmetic::divideIntsToFloat( m_framesPerSecondDenominator, m_framesPerSecondNumerator );
}

int MediaFoundationOutputVideoStream::bitsPerSecond() const
{
	return m_bitsPerSecond;
}

int MediaFoundationOutputVideoStream::currentFrameCount() const
{
	return m_frameIndex;
}

int64 MediaFoundationOutputVideoStream::currentTime() const
{
	return m_currentTime;
}

float MediaFoundationOutputVideoStream::currentTimeMilliseconds() const
{	
	return m_frameIndex * frameDurationMilliseconds();
}

bool MediaFoundationOutputVideoStream::appendFrameRGBA( Array2DView< ubyte4 > rgba )
{
	if( !m_valid )
	{
		return false;
	}

	if( size() != rgba.size() )
	{
		return false;
	}

	Swizzle::RGBAToBGRA( rgba, m_bgraData );
	return appendFrameBGRA( m_bgraData );
}

bool MediaFoundationOutputVideoStream::appendFrameBGRA( Array2DView< ubyte4 > bgra )
{
	if( !m_valid )
	{
		return false;
	}

	if( size() != bgra.size() )
	{
		return false;
	}

	if( m_codec == H264 )
	{
		bgra = ArrayUtils::flippedUpDownView< ubyte4 >( bgra );
	}

	// copy data to buffer
	BYTE* pBufferData = nullptr;
	HRESULT hr = m_pBuffer->Lock( &pBufferData, nullptr, nullptr );

	Array2DView< ubyte4 > srcView;	
	if( bgra.elementsArePacked() )
	{
		srcView = bgra;
	}
	else
	{
		ArrayUtils::copy< ubyte4 >( bgra, m_bgraData );
		srcView = m_bgraData;
	}

	if( SUCCEEDED( hr ) )
	{
		DWORD dstRowPitch = 4 * m_width;

		hr = MFCopyImage
		(
			pBufferData, // destination pointer
			dstRowPitch, // destination row pitch
			reinterpret_cast< BYTE* >( srcView.rowPointer( 0 ) ), // first row in source image.
			srcView.rowPitchBytes(), // source row pitch			

			srcView.width() * sizeof( ubyte4 ), // source width in bytes
			srcView.height() // source height
		);
	}

	if( pBufferData != nullptr )
	{
		m_pBuffer->Unlock();
	}

	// set the time stamp and the duration.
	if( SUCCEEDED( hr ) )
	{
		hr = m_pSample->SetSampleTime( m_currentTime );
	}
	if( SUCCEEDED( hr ) )
	{
		hr = m_pSample->SetSampleDuration( m_sampleDuration );
	}

	// send the sample to the sink writer
	if( SUCCEEDED( hr ) )
	{
		hr = m_pSinkWriter->WriteSample( m_streamIndex, m_pSample );
	}

	++m_frameIndex;
	m_currentTime += m_sampleDuration;

	return SUCCEEDED( hr );
}

bool MediaFoundationOutputVideoStream::close()
{
	m_valid = false;

	HRESULT hr = m_pSinkWriter->Finalize();
	return SUCCEEDED( hr );
}

MediaFoundationOutputVideoStream::MediaFoundationOutputVideoStream(
	QString filename,
	Codec codec,
	int width, int height,
	int framesPerSecondNumerator, int framesPerSecondDenominator,
	int bitsPerSecond ) :

	m_valid( false ),

	m_codec( codec ),

	m_width( width ),
	m_height( height ),

	m_framesPerSecondNumerator( framesPerSecondNumerator ),
	m_framesPerSecondDenominator( framesPerSecondDenominator ),
	m_bitsPerSecond( bitsPerSecond ),

	m_frameIndex( 0 ),
	m_currentTime( 0 ),

	m_pSinkWriter( nullptr ),
	m_pMediaTypeOut( nullptr ),
	m_pMediaTypeIn( nullptr ),
	m_pSample( nullptr ),
	m_pBuffer( nullptr ),

	m_bgraData( width, height )

{
	MFFrameRateToAverageTimePerFrame( framesPerSecondNumerator, framesPerSecondDenominator, &m_sampleDuration );

	HRESULT hr = MFCreateSinkWriterFromURL( filename.utf16(), nullptr, nullptr, &m_pSinkWriter );

	if( SUCCEEDED( hr ) )
	{
		hr = initializeOutputType();
	}

	if( SUCCEEDED( hr ) )
	{
		hr = initializeInputType();
	}

	if( SUCCEEDED( hr ) )
	{
		hr = initializeBuffer();
	}

	// tell the sink writer to start accepting data
	if( SUCCEEDED( hr ) )
	{
		hr = m_pSinkWriter->BeginWriting();
	}

	if( SUCCEEDED( hr ) )
	{
		m_valid = true;
	}
}


HRESULT MediaFoundationOutputVideoStream::initializeOutputType()
{
	GUID format;
	switch( m_codec )
	{
	case H264:
		format = MFVideoFormat_H264;
		break;

	case VC1:
		format = MFVideoFormat_WVC1;
		break;

	case WMV9:
		format = MFVideoFormat_WMV3;
		break;

	default:
		format = MFVideoFormat_H264;
		break;
	}
	
	// create the output video type
	// and set its attributes
	HRESULT hr = MFCreateMediaType( &m_pMediaTypeOut );

	if( SUCCEEDED( hr ) )
	{
		hr = m_pMediaTypeOut->SetGUID( MF_MT_MAJOR_TYPE, MFMediaType_Video );
	}
	if( SUCCEEDED( hr ) )
	{
		hr = m_pMediaTypeOut->SetGUID( MF_MT_SUBTYPE, format );
	}
	if( SUCCEEDED( hr ) )
	{
		hr = m_pMediaTypeOut->SetUINT32( MF_MT_AVG_BITRATE, m_bitsPerSecond );
	}
	if( SUCCEEDED( hr ) )
	{
		hr = m_pMediaTypeOut->SetUINT32( MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive );
	}
	if( SUCCEEDED( hr ) )
	{
		hr = MFSetAttributeSize( m_pMediaTypeOut, MF_MT_FRAME_SIZE, m_width, m_height );   
	}
	if( SUCCEEDED( hr ) )
	{
		hr = MFSetAttributeRatio( m_pMediaTypeOut, MF_MT_FRAME_RATE, m_framesPerSecondNumerator, m_framesPerSecondDenominator );
	}
	if( SUCCEEDED( hr ) )
	{
		hr = MFSetAttributeRatio( m_pMediaTypeOut, MF_MT_PIXEL_ASPECT_RATIO, 1, 1 );
	}
	if( SUCCEEDED( hr ) )
	{
		hr = m_pSinkWriter->AddStream( m_pMediaTypeOut, &m_streamIndex );
	}

	return hr;
}

HRESULT MediaFoundationOutputVideoStream::initializeInputType()
{
	// create the input video type
	// and set its attributes
	HRESULT hr = MFCreateMediaType( &m_pMediaTypeIn );   

	if( SUCCEEDED( hr ) )
	{
		hr = m_pMediaTypeIn->SetGUID( MF_MT_MAJOR_TYPE, MFMediaType_Video );   
	}
	if( SUCCEEDED( hr ) )
	{
		hr = m_pMediaTypeIn->SetGUID( MF_MT_SUBTYPE, MFVideoFormat_RGB32 );     
	}
	if( SUCCEEDED( hr ) )
	{
		hr = m_pMediaTypeIn->SetUINT32( MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive );   
	}
	if( SUCCEEDED( hr ) )
	{
		hr = MFSetAttributeSize( m_pMediaTypeIn, MF_MT_FRAME_SIZE, m_width, m_height );   
	}
	if( SUCCEEDED( hr ) )
	{
		hr = MFSetAttributeRatio( m_pMediaTypeIn, MF_MT_FRAME_RATE, m_framesPerSecondNumerator, m_framesPerSecondDenominator );
	}
	if( SUCCEEDED( hr ) )
	{
		hr = MFSetAttributeRatio( m_pMediaTypeIn, MF_MT_PIXEL_ASPECT_RATIO, 1, 1 );   
	}
	if( SUCCEEDED( hr ) )
	{
		hr = m_pSinkWriter->SetInputMediaType( m_streamIndex, m_pMediaTypeIn, nullptr );
	}

	return hr;
}

HRESULT MediaFoundationOutputVideoStream::initializeBuffer()
{
	// Create a new memory buffer
	int nBytes = 4 * m_width * m_height;
	HRESULT hr = MFCreateMemoryBuffer( nBytes, &m_pBuffer );

	// Set the data length of the buffer.
	if( SUCCEEDED( hr ) )
	{
		hr = m_pBuffer->SetCurrentLength( nBytes );
	}

	// Create a media sample and add the buffer to the sample.
	if( SUCCEEDED( hr ) )
	{
		hr = MFCreateSample( &( m_pSample ) );
	}
	if( SUCCEEDED( hr ) )
	{
		hr = m_pSample->AddBuffer( m_pBuffer );
	}

	return hr;
}
