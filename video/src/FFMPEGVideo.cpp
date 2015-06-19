#include "FFMPEGVideo.h"

#include <cassert>
#include <vector>

#include "math/Arithmetic.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
FFMPEGVideo* FFMPEGVideo::fromFile( const char* filename )
{
    // one time registration
    if( !( FFMPEGVideo::s_bInitialized ) )
    {
        // register all the muxers, demuxers and protocols.
        av_register_all();
        // register all codecs.
        avcodec_register_all();
        FFMPEGVideo::s_bInitialized = true;
    }

    int retVal;

    AVFormatContext* pFormatContext = avformat_alloc_context();
    AVCodecContext* pCodecContext = NULL;
    AVCodec* pCodec = NULL;
    AVFrame* pFrameRaw = NULL;
    AVFrame* pFrameRGB = NULL;
    std::vector< SwsContext* > contexts;

    // Open the file and examine the header
    // populates pFormatContext
    retVal = avformat_open_input
    (
        &pFormatContext, // output context
        filename, // filename
        NULL, // format, NULL --> auto-detect, otherwise, forces file format
        NULL // format options, NULL --> auto-detect, otherwise force decode options
    );

    if( retVal == 0 ) // if succeeded
    {
        // Retrieve stream information
        // populates pFormatContext->streams with data

        // NULL: don't want output dictionary
        retVal = avformat_find_stream_info( pFormatContext, NULL );
        if( retVal >= 0 ) // if succeeded
        {
            // TODO: let the user select which stream
            // Find the first video stream
            uint32_t i = 0;
            int videoStreamIndex = -1;
            while( ( videoStreamIndex == -1 ) && ( i < pFormatContext->nb_streams ) )
            {
                if( pFormatContext->streams[ i ]->codec->codec_type == AVMEDIA_TYPE_VIDEO )
                {
                    videoStreamIndex = i;
                }
                ++i;
            }

            // if we found a video stream
            // load its codec context
            if( videoStreamIndex > -1 )
            {
                // get a pointer to the codec context for the video stream
                pCodecContext = pFormatContext->streams[ videoStreamIndex ]->codec;

                // find a codec for the codec context
                pCodec = avcodec_find_decoder( pCodecContext->codec_id );
                if( pCodec != NULL )
                {
                    // ok we found a codec, try opening it
                    // NULL: don't want output dictionary of codec-private options.
                    retVal = avcodec_open2( pCodecContext, pCodec, NULL );
                    if( retVal >= 0 )
                    {
                        // Allocate a frame for the incoming data	                        
                        pFrameRaw = av_frame_alloc();
                        if( pFrameRaw != NULL )
                        {
                            // Allocate another for RGB
                            pFrameRGB = av_frame_alloc();
                            if( pFrameRGB != NULL )
                            {
                                std::vector< SwsContext* > contexts = createSWSContexts( pCodecContext );
                                if( contexts.size() > 0 )
                                {
                                    FFMPEGVideo* pOutput = new FFMPEGVideo
                                    (
                                        pFormatContext,
                                        videoStreamIndex,
                                        pCodecContext,
                                        pFrameRaw,
                                        pFrameRGB,
                                        contexts
                                    );

                                    if( pOutput != nullptr )
                                    {
                                        return pOutput;
                                    }
                                    else
                                    {
                                        fprintf( stderr, "Out of memory allocating video object!\n" );
                                    }

                                    for( SwsContext* pContext : contexts )
                                    {
                                        sws_freeContext( pContext );
                                    }
                                }
                                else
                                {
                                    fprintf( stderr, "Error creating RGB conversion context!\n" );
                                }

                                av_free( pFrameRGB );
                            }
                            else
                            {
                                fprintf( stderr, "Error allocating RGB frame!\n" );
                            }

                            av_free( pFrameRaw );
                        }
                    }
                    else
                    {
                        fprintf( stderr, "Error opening codec!\n" );
                    }
                }
                else
                {
                    fprintf( stderr, "Unsupported codec!\n" );
                }
            }
            else
            {
                fprintf( stderr, "File contains no video streams!\n" );
            }
        }
        else
        {
            fprintf( stderr, "Error parsing stream information!\n" );
        }

        // close the video file in case of failure
        avformat_close_input( &pFormatContext );
    }
    else
    {
        fprintf( stderr, "Error opening %s!\n", filename );
    }

    assert( false );
    return NULL;
}

// virtual
FFMPEGVideo::~FFMPEGVideo()
{
    for( SwsContext* pContext : m_swsContexts )
    {
        sws_freeContext( pContext );
    }
    av_free( m_pFrameDecoded );
    av_free( m_pFrameRaw );
    avcodec_close( m_pCodecContext );
    avformat_close_input( &m_pFormatContext );
}

// virtual
int64_t FFMPEGVideo::numFrames() const
{
    return m_nFrames;
}

// virtual
float FFMPEGVideo::durationMilliseconds() const
{
    return 1000.0f * durationSeconds();
}

// virtual
float FFMPEGVideo::durationSeconds() const
{
    return m_durationSeconds;
}

// virtual
int64_t FFMPEGVideo::durationRationalTimePeriods() const
{
    return m_durationRationalTimePeriods;
}

// virtual
float FFMPEGVideo::framePeriodMilliseconds() const
{
    return( 1000.f * framePeriodSeconds() );
}

// virtual
float FFMPEGVideo::framePeriodSeconds() const
{
    return m_framePeriodSeconds;
}

// virtual
Vector2i FFMPEGVideo::framePeriodRationalSeconds() const
{
    return m_framePeriodRationalSeconds;
}

// virtual
int FFMPEGVideo::width() const
{
    return m_width;
}

// virtual
int FFMPEGVideo::height() const
{
    return m_height;
}

// virtual
Vector2i FFMPEGVideo::size() const
{
    return{ m_width, m_height };
}

// virtual
int FFMPEGVideo::bytesPerFrame() const
{
    return m_nBytesPerFrame;
}

// virtual
int64_t FFMPEGVideo::getNextFrameIndex() const
{
    return m_nextFrameIndex;
}

// virtual
bool FFMPEGVideo::setNextFrameIndex( int64_t frameIndex )
{
    // if frameIndex is out of range, then return false
    if( frameIndex < 0 || frameIndex >= m_nFrames )
    {
#if _WIN32		
        fprintf( stderr, "Cannot seek to frame %I64d, frameIndex must be between 0 and %I64d\n", frameIndex, m_nFrames );
#else
        fprintf( stderr, "Cannot seek to frame %lld, frameIndex must be between 0 and %lld\n", frameIndex, m_nFrames );
#endif
        return false;
    }

    // else if it's going to be the next frame anyway
    // then do nothing
    if( frameIndex == m_nextFrameIndex )
    {
        return true;
    }

    // TODO: When do I use AVSEEK_FLAG_BACKWARD?
    // TODO: also assumes a frame is one tick
    int64_t frameDuration = m_durationRationalTimePeriods / m_nFrames; // in codec time base ticks
    int seekFlags = AVSEEK_FLAG_ANY;

    // tell ffmpeg to seek
    int64_t frameTimestamp = frameIndex * frameDuration;
    int retVal = av_seek_frame( m_pFormatContext, m_videoStreamIndex, frameTimestamp, seekFlags );

    // seek to the nearest keyframe
    //int seekFlags = AVSEEK_FLAG_FRAME;

    // tell ffmpeg to seek
    // TODO: switch to the new ffmpeg seek API once it works
    // TODO: this only works if it's a MJPEG sequence...
    //int retVal = avformat_seek_file( m_pFormatContext, m_videoStreamIndex, frameIndex, frameIndex, frameIndex, seekFlags );
    if( retVal < 0 )
    {
#if _WIN32		
        fprintf( stderr, "ffmpeg error seeking to frame: %I64d\n", frameIndex );
#else
        fprintf( stderr, "ffmpeg error seeking to frame: %lld\n", frameIndex );
#endif
        return false;
    }

    // seek was successful, flush codec internal buffers
    avcodec_flush_buffers( m_pCodecContext );
    m_nextFrameIndex = frameIndex;
    return true;
}

// TODO: refactor
// virtual
bool FFMPEGVideo::getNextFrameRGB24( Array2DView< uint8x3 > rgbOut )
{
    if( m_nextFrameIndex >= m_nFrames )
    {
        return false;
    }

    // TODO: can potentially accelerate this by using m_pCodecContext->hurry_up = 1
    int64_t t;

    bool decodeSucceeded = decodeNextFrame( &t );
    while( decodeSucceeded && ( t < m_nextFrameIndex ) )
    {
        decodeSucceeded = decodeNextFrame( &t );
    }

    // if the loop was successful
    // then t = m_nextFrameIndex
    if( decodeSucceeded )
    {
        // convert the decoded frame to RGB
        convertDecodedFrameToRGB( rgbOut );

        ++m_nextFrameIndex;
        return true;
    }
    else
    {
        return false;
    }
}

// virtual
bool FFMPEGVideo::getNextFrameBGRA32( Array2DView< uint8x4 > bgraOut )
{
    if( m_nextFrameIndex >= m_nFrames )
    {
        return false;
    }

    // TODO: can potentially accelerate this by using m_pCodecContext->hurry_up = 1
    int64_t t;

    bool decodeSucceeded = decodeNextFrame( &t );
    while( decodeSucceeded && ( t < m_nextFrameIndex ) )
    {
        decodeSucceeded = decodeNextFrame( &t );
    }

    // if the loop was successful
    // then t = m_nextFrameIndex
    if( decodeSucceeded )
    {
        // Associate the output buffer with m_pFrameDecoded.
        avpicture_fill( ( AVPicture* )m_pFrameDecoded,
            reinterpret_cast< const uint8_t* >( bgraOut.pointer() ),
            PIX_FMT_BGRA,
            width(), height() );

        // Software scale the raw frame to m_pFrameDecoded, which is associated with rgbOut.
        sws_scale( m_swsContexts[ 1 ], // converter
            m_pFrameRaw->data, m_pFrameRaw->linesize, // source data and stride
            0, height(), // starting y and height
            m_pFrameDecoded->data, m_pFrameDecoded->linesize
        );

        ++m_nextFrameIndex;
        return true;
    }
    else
    {
        return false;
    }
}

//////////////////////////////////////////////////////////////////////////
// Private
//////////////////////////////////////////////////////////////////////////

FFMPEGVideo::FFMPEGVideo( AVFormatContext* pFormatContext, int videoStreamIndex,
    AVCodecContext* pCodecContext,
    AVFrame* pFrameRaw,
    AVFrame* pFrameRGB,
    const std::vector< SwsContext* > & swsContexts ) :

    m_pFormatContext( pFormatContext ),
    m_videoStreamIndex( videoStreamIndex ),
    m_pCodecContext( pCodecContext ),
    m_pFrameRaw( pFrameRaw ),
    m_pFrameDecoded( pFrameRGB ),
    m_swsContexts( swsContexts ),

    m_width( pCodecContext->width ),
    m_height( pCodecContext->height ),

    m_nextFrameIndex( 0 )

{
    m_nFrames = m_pFormatContext->streams[ m_videoStreamIndex ]->nb_frames;
    
    m_nBytesPerFrame = avpicture_get_size( PIX_FMT_RGB24, width(), height() );

    AVRational averageFrameRate = m_pFormatContext->streams[ m_videoStreamIndex ]->avg_frame_rate;
    AVRational averageFramePeriod = av_inv_q( averageFrameRate );
    m_framePeriodRationalSeconds.x = averageFramePeriod.num;
    m_framePeriodRationalSeconds.y = averageFramePeriod.den;
    m_framePeriodSeconds = static_cast< float >( av_q2d( averageFramePeriod ) );

    AVRational timeBaseRationalSeconds = m_pFormatContext->streams[ m_videoStreamIndex ]->time_base;

    m_durationRationalTimePeriods = m_pFormatContext->streams[ m_videoStreamIndex ]->duration;
    m_durationSeconds = static_cast< float >( m_durationRationalTimePeriods * av_q2d( timeBaseRationalSeconds ) );
}

bool FFMPEGVideo::decodeNextFrame( int64_t* decodedFrameIndex )
{
    AVPacket packet;
    bool readFrameSucceeded;
    bool decodeSucceeded = true;

    int frameFinished = 0; // frameFinished > 0 means the frame is finished
    readFrameSucceeded = ( av_read_frame( m_pFormatContext, &packet ) >= 0 );
    // loop while it can still read (bSucceeded) and
    // the frame is NOT done (frameFinished == 0)
    while( readFrameSucceeded && ( frameFinished == 0 ) )
    {
        // printf( "decodeNextFrame: packet.dts = %I64d, packet.pts = %I64d\n", packet.dts, packet.pts );

        // is this a packet from the video stream we selected?
        if( packet.stream_index == m_videoStreamIndex )
        {
            // if so, then decode it
            decodeSucceeded = ( avcodec_decode_video2( m_pCodecContext, m_pFrameRaw,
                &frameFinished, &packet ) > 0 );

            // we failed in decoding the video
            if( !decodeSucceeded )
            {
#if _WIN32
                fprintf( stderr, "ffmpeg error decoding video frame: %I64d\n", m_nextFrameIndex );
#else
                fprintf( stderr, "ffmpeg error decoding video frame: %lld\n", m_nextFrameIndex );
#endif
                // always free the packet that was allocated by av_read_frame
                av_free_packet( &packet );
                return false;
            }

            if( decodedFrameIndex != NULL )
            {
                // HACK
                // TODO: fix this
                int64_t frameDuration = m_durationRationalTimePeriods / m_nFrames; // in codec time base ticks
                *decodedFrameIndex = packet.pts / frameDuration; // ffmpeg uses 0-based frame indices
            }
        }

        // always free the packet that was allocated by av_read_frame
        av_free_packet( &packet );

        // if the frame isn't finished, then read another packet
        if( frameFinished == 0 )
        {
            readFrameSucceeded = ( av_read_frame( m_pFormatContext, &packet ) >= 0 );
        }
    };

    if( !readFrameSucceeded )
    {
#if _WIN32
        fprintf( stderr, "ffmpeg error reading next packet on frame: %I64d\n", m_nextFrameIndex );
#else
        fprintf( stderr, "ffmpeg error reading next packet on frame: %lld\n", m_nextFrameIndex );
#endif
        return false;
    }
    return true;
}

void FFMPEGVideo::convertDecodedFrameToRGB( Array2DView< uint8x3 > rgbOut )
{
    // Associate the output buffer with m_pFrameDecoded.
    avpicture_fill( ( AVPicture* )m_pFrameDecoded,
        reinterpret_cast< const uint8_t* >( rgbOut.pointer() ),
        PIX_FMT_RGB24,
        width(), height() );

    // Software scale the raw frame to m_pFrameDecoded, which is associated with rgbOut.
    sws_scale
    (
        m_swsContexts[ 0 ], // converter
        m_pFrameRaw->data, m_pFrameRaw->linesize, // source data and stride
        0, height(), // starting y and height
        m_pFrameDecoded->data, m_pFrameDecoded->linesize
    );
}

bool FFMPEGVideo::isDecodedFrameKey()
{
    return( m_pFrameRaw->key_frame == 1 );
}

std::vector< SwsContext* > FFMPEGVideo::createSWSContexts( AVCodecContext* pCodecContext )
{
    // Note: PixelFormats are in avutil.h
    // Note: flags are in swscale.h

    SwsContext* pRGB24 = sws_getContext
    (
        pCodecContext->width, pCodecContext->height, // source width and height
        pCodecContext->pix_fmt, // source format
        pCodecContext->width, pCodecContext->height, // destination width and height
        PIX_FMT_RGB24, // destination format
        SWS_POINT, // flags
        NULL, // source filter, NULL --> default
        NULL, // destination filter, NULL --> default
        NULL // filter parameters, NULL --> default
    );

    SwsContext* pBGRA32 = sws_getContext
    (
        pCodecContext->width, pCodecContext->height, // source width and height
        pCodecContext->pix_fmt, // source format
        pCodecContext->width, pCodecContext->height, // destination width and height
        PIX_FMT_BGRA, // destination format
        SWS_POINT, // flags
        NULL, // source filter, NULL --> default
        NULL, // destination filter, NULL --> default
        NULL // filter parameters, NULL --> default
    );

    // TODO: check for NULL.
    return{ pRGB24, pBGRA32 };
}

// static
bool FFMPEGVideo::s_bInitialized = false;
