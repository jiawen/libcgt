#pragma once

#include <cstdint>
#include <vector>

#include <vecmath/Vector2i.h>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#include "IVideo.h"

// A video loaded using ffmpeg implementing the IVideo interface
class FFMPEGVideo
{
public:

	static FFMPEGVideo* fromFile( const char* filename );
	virtual ~FFMPEGVideo();

    // The number of frames, if the file has it.
    // Otherwise, returns 0.
    virtual int64_t numFrames() const;
    
    virtual float durationMilliseconds() const;
    virtual float durationSeconds() const;
    // Multiply this by framePeriodRationalSeconds() to get the duration
    // in seconds as a rational. If it's not specified directly in the file,
    // then it will be estimated based on file size / bitrate.
    virtual int64_t durationRationalTimePeriods() const;

    virtual float framePeriodMilliseconds() const;
    virtual float framePeriodSeconds() const;
    // The "timebase" in seconds as a rational number of two 32-bit integers.
    virtual Vector2i framePeriodRationalSeconds() const;

    virtual int width() const;
    virtual int height() const;
    virtual Vector2i size() const; // (width, height)
	virtual int bytesPerFrame() const; // TODO: get rid of this? this assumes PIX_FMT_RGB24

	// Returns the internal frame counter.
    virtual int64_t getNextFrameIndex() const;
    // Seek to time frameIndex.
    virtual bool setNextFrameIndex( int64_t frameIndex );

	// Populates rgbOut with the contents of the next frame
	// *and increments* the internal frame counter
	// returns true if succeeded
	// and false on failure (i.e. at the end of the video stream).
    // TODO: check that rgbOut has the right width, height and is packed.
    // TODO: other formats are easy, bgr, etc
	virtual bool getNextFrameRGB24( Array2DView< uint8x3 > rgbOut );

    virtual bool getNextFrameBGRA32( Array2DView< uint8x4 > bgraOut );

private:

    FFMPEGVideo( AVFormatContext* pFormatContext, int iVideoStreamIndex,
        AVCodecContext* pCodecContext,
        AVFrame* pFrameRaw,
        AVFrame* pFrameRGB,
        const std::vector< SwsContext* > & swsContexts );

    static std::vector< SwsContext* > createSWSContexts( AVCodecContext* pContext );

	// reads the next frame in its internal format and stores it in m_pFrameRaw
	// on success, returns the index of the frame that was decoded
	// returns -1 on failure
	bool decodeNextFrame( int64_t* decodedFrameIndex );

	// converts the next frame into RGB
    void convertDecodedFrameToRGB( Array2DView< uint8x3 > rgbOut );

	bool isDecodedFrameKey();

	// initially false
	// set to true once global ffmpeg initialization is complete
	// (initialized the first time an FFMPEGVideo is created)
	static bool s_bInitialized;

	AVFormatContext* m_pFormatContext;
	int m_videoStreamIndex;
    AVCodecContext* m_pCodecContext;
	AVFrame* m_pFrameRaw;
    AVFrame* m_pFrameDecoded;
    std::vector< SwsContext* > m_swsContexts; // for YUV --> various formats

	// dimensions
	int m_width;
	int m_height;
	int64_t m_nFrames;

    // time base
    Vector2i m_framePeriodRationalSeconds;

    // duration in time base units
    int64_t m_durationRationalTimePeriods;
    // duration in seconds
    float m_durationSeconds;

	// derived units
	int m_nBytesPerFrame;
	float m_framePeriodSeconds;

	int64_t m_nextFrameIndex;
};
