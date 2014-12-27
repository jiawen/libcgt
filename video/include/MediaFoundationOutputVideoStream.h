#pragma once

#include <windows.h>
#include <mfidl.h>
#include <mfapi.h>
#include <mfreadwrite.h>
#include <mferror.h>

#include <common/Array2D.h>
#include <common/Array2DView.h>

class QString;

class MediaFoundationOutputVideoStream
{
public:

	enum Codec
	{
		H264,
		VC1,
		WMV9
	};

	static int recommendedBitsPerSecond( int width, int height, int framesPerSecondNumerator = 24,
		int framesPerSecondDenominator = 1 );

	// opens a new video file for writing
	//
	// framesPerSecond = framesPerSecondNumerator / framesPerSecondDenominator
	// i.e., 24 fps --> (24,1), 29.97 fps -> (30000,1001)
	//
	// returns nullptr if the framework can't open the file
	// or the combination of parameters 
	static MediaFoundationOutputVideoStream* open
	(
		QString filename,
		Codec codec,
		int width, int height,
		int framesPerSecondNumerator, int framesPerSecondDenominator,
		int bitsPerSecond
	);

	virtual ~MediaFoundationOutputVideoStream();

	Codec codec() const;

	int width() const;
	int height() const;
	Vector2i size() const;

	// framerate
	float framesPerSecond() const;
	Vector2i framesPerSecondRational() const;
	
	// returns the duration of a frame in 100 ns intervals
	int64_t frameDuration() const;

	// returns the duration of a frame in milliseconds
	float frameDurationMilliseconds() const;

	// encoding bitrate
	int bitsPerSecond() const;

	// returns the number of frames appended
	int currentFrameCount() const;

	// returns the number of 100-nanosecond intervals appended
	int64_t currentTime() const;
	
	// returns the number of milliseconds appended
	float currentTimeMilliseconds() const;

	// appends a frame to be encoded in either format
    bool appendFrameRGBA( Array2DView< uint8x4 > rgba );
    bool appendFrameBGRA( Array2DView< uint8x4 > bgra );

	// flushes the pipeline and writes everything to disk
	// this stream is no longer usable afterwards
	bool close();

private:

	MediaFoundationOutputVideoStream
	(
		QString filename,
		Codec codec,
		int width, int height,
		int framesPerSecondNumerator, int framesPerSecondDenominator,
		int bitsPerSecond
	);

	HRESULT initializeOutputType();
	HRESULT initializeInputType();
	HRESULT initializeBuffer();

	bool m_valid;

	Codec m_codec;

	int m_width;
	int m_height;
	int m_framesPerSecondNumerator;
	int m_framesPerSecondDenominator;
	int m_bitsPerSecond;

	UINT64 m_sampleDuration; // in units of 100-nanosecond intervals

	IMFSinkWriter* m_pSinkWriter;
		IMFMediaType* m_pMediaTypeOut;
		IMFMediaType* m_pMediaTypeIn;
		DWORD m_streamIndex; // should be 0

	IMFSample* m_pSample;
		IMFMediaBuffer* m_pBuffer;

	// tracking the current frame index
	// and associated time
	int m_frameIndex;
	LONGLONG m_currentTime;

	// in case we need to swizzle
    Array2D< uint8x4 > m_bgraData;
};
