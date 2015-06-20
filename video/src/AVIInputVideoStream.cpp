#include "AVIInputVideoStream.h"

#include <common/ArrayUtils.h>
#include <imageproc/Swizzle.h>

// static
AVIInputVideoStream* AVIInputVideoStream::open( PAVIFILE pAVIFile, int streamIndex,
    int width, int height )
{
    PAVISTREAM pStream;
    HRESULT hr = AVIFileGetStream( pAVIFile, &pStream, streamtypeVIDEO, streamIndex );

    if( FAILED( hr ) )
    {
        return nullptr;
    }

    int streamStartIndex = AVIStreamStart( pStream );
    int nSamples = AVIStreamLength( pStream );

    // grab the stream info
    AVISTREAMINFO streamInfo;
    AVIStreamInfo( pStream, &streamInfo, sizeof( AVISTREAMINFO ) );

    // open a frame for decompression
    BITMAPINFOHEADER bih;
    ZeroMemory( &bih, sizeof( bih ) );

    bih.biSize = sizeof( BITMAPINFOHEADER );
    bih.biWidth = width;
    bih.biHeight = height;
    bih.biPlanes = 1;
    bih.biBitCount = 24; // 32 doesn't work
    bih.biCompression = BI_RGB;
    bih.biSizeImage = 0; // 0 for BI_RGB, otherwise the size of the buffer
    bih.biXPelsPerMeter = 0; // don't care
    bih.biYPelsPerMeter = 0; // don't care
    bih.biClrUsed = 0; // color index
    bih.biClrImportant = 0; // color index

    PGETFRAME pFrame = AVIStreamGetFrameOpen( pStream, &bih  );
    //PGETFRAME pFrame = AVIStreamGetFrameOpen( pStream, (BITMAPINFOHEADER*) AVIGETFRAMEF_BESTDISPLAYFMT ); // only for displaying to screen
    //PGETFRAME pFrame = AVIStreamGetFrameOpen( pStream, NULL ); // doesn't work

    if( pFrame == NULL )
    {
        AVIStreamRelease( pStream );
        return nullptr;
    }

    AVIInputVideoStream* pOutput = new AVIInputVideoStream;

    pOutput->m_ubyte4Data.resize( { width, height } );
    pOutput->m_width = width;
    pOutput->m_height = height;

    pOutput->m_pStream = pStream;
    pOutput->m_pFrame = pFrame;
    pOutput->m_info = streamInfo;

    pOutput->m_streamStartIndex = streamStartIndex;
    pOutput->m_nSamples = nSamples;

    return pOutput;
}

// virtual
AVIInputVideoStream::~AVIInputVideoStream()
{
    AVIStreamGetFrameClose( m_pFrame );
    AVIStreamRelease( m_pStream );
}

int AVIInputVideoStream::width() const
{
    return m_width;
}

int AVIInputVideoStream::height() const
{
    return m_height;
}

Vector2i AVIInputVideoStream::size() const
{
    return{ width(), height() };
}

int AVIInputVideoStream::numFrames() const
{
    return m_nSamples;
}

float AVIInputVideoStream::framesPerSecond() const
{
    return Arithmetic::divideIntsToFloat( m_info.dwRate, m_info.dwScale );
}

void AVIInputVideoStream::framesPerSecondRational( int& numerator, int& denominator ) const
{
    numerator = m_info.dwRate;
    denominator = m_info.dwScale;
}

Array2DView< uint8x4 > AVIInputVideoStream::getFrameRGBA( int frameIndex, uint8_t alpha )
{
    if( frameIndex < 0 || frameIndex >= numFrames() )
    {
        return Array2DView< uint8x4 >();
    }

    Array2DView< uint8x3 > bgrData = getFrameBGR( frameIndex );
    libcgt::core::imageproc::swizzle::BGRToRGBA( bgrData, m_ubyte4Data, alpha );
    return m_ubyte4Data;
}

Array2DView< uint8x4 > AVIInputVideoStream::getFrameBGRA( int frameIndex, uint8_t alpha )
{
    if( frameIndex < 0 || frameIndex >= numFrames() )
    {
        return Array2DView< uint8x4 >();
    }

    Array2DView< uint8x3 > bgrData = getFrameBGR( frameIndex );
    libcgt::core::imageproc::swizzle::BGRToBGRA( bgrData, m_ubyte4Data, alpha );
    return m_ubyte4Data;
}

Array2DView< uint8x3 > AVIInputVideoStream::getFrameBGR( int frameIndex )
{
    if( frameIndex < 0 || frameIndex >= numFrames() )
    {
        return Array2DView< uint8x3 >( );
    }

    // grab the frame
    int streamFrameIndex = frameIndex - m_streamStartIndex;
    BITMAPINFOHEADER* pBitmapData = reinterpret_cast< BITMAPINFOHEADER* >( AVIStreamGetFrame( m_pFrame, streamFrameIndex ) );

    // data is just past the header
    uint8_t* pBGRDataBegin = reinterpret_cast< uint8_t* >( pBitmapData )+sizeof( BITMAPINFOHEADER )+pBitmapData->biClrUsed * sizeof( RGBQUAD );

    // the original data is stored bottom to top
    Array2DView< uint8x3 > inputView( pBGRDataBegin, { m_width, m_height } );
    return ArrayUtils::flippedUpDownView< uint8x3 >( inputView );
}

AVIInputVideoStream::AVIInputVideoStream()
{

}
