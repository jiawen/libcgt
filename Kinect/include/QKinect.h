#pragma once

#include <QObject>
#include <QVector>

// In Visual Studio 2010
#if ( _MSC_VER <= 1600 )
#pragma warning( disable: 4005 ) // stdint.h and intsafe.h: disable warnings for multiply defined
#pragma warning( disable: 4805 ) // sphelper.h: disable unsafe mix of BOOL and bool
#endif

#include <windows.h>
#include <NuiApi.h>

// For configuring DMO properties
#include <wmcodecdsp.h>

// For discovering microphone array device
#include <MMDeviceApi.h>
#include <devicetopology.h>

// For functions and definitions used to create output file
#include <uuids.h> // FORMAT_WaveFormatEx and such
#include <mfapi.h> // FCC

// For speech APIs
#include <sapi.h>
#include <sphelper.h>

#include <map>
#include <vector>

#include <common/Array2D.h>
#include <imageproc/Image4ub.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

#include "KinectStream.h"

class QKinect : public QObject
{
	Q_OBJECT

public:

	static int numDevices();

	// Creates a Kinect device
	// deviceIndex should be between [0, numDevices()),
	// nuiFlags is an bit mask combination of:
	// NUI_INITIALIZE_FLAG_USES_COLOR,
	// NUI_INITIALIZE_FLAG_USES_DEPTH,
	// NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX,
	// NUI_INITIALIZE_FLAG_USES_SKELETON,
	// and NUI_INITIALIZE_FLAG_USES_AUDIO

	// NUI_INITIALIZE_FLAG_USES_DEPTH or NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX must be set if you want to recognize skeletons with NUI_INITIALIZE_FLAG_USES_SKELETON
	// --> NUI_INITIALIZE_FLAG_USES_SKELETON must be set to use NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX
	// NUI_INITIALIZE_FLAG_USES_DEPTH or NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX must be set if you want to enable near mode
	// NUI_INITIALIZE_FLAG_USES_AUDIO must be set if you want to recognize any of the phrases in recognizedPhrases
	static QKinect* create
	(
		int deviceIndex = 0,
		DWORD nuiFlags = NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH,		
		NUI_IMAGE_RESOLUTION rgbResolution = NUI_IMAGE_RESOLUTION_640x480,
		NUI_IMAGE_RESOLUTION depthResolution = NUI_IMAGE_RESOLUTION_640x480,
		bool useExtendedDepth = true,
		QVector< QString > recognizedPhrases = QVector< QString >()
	);

	// returns a vector of pairs of indices (i,j)
	// such that within a NUI_SKELETON_FRAME
	// frame.SkeletonData[k].SkeletonPositions[i] --> frame.SkeletonData[k].SkeletonPositions[j] is a bone
	static const std::vector< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX > >& jointIndicesForBones();

	// returns the inverse of jointIndicesForBones
	static const std::map< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX >, int >& boneIndicesForJoints();

	enum QKinectEvent
	{
		QKinect_Event_Failed,
		QKinect_Event_Timeout,
		QKinect_Event_Skeleton,
		QKinect_Event_RGB,
		QKinect_Event_Depth
	};

	enum QKinectBone
	{
		QKINECT_LOWER_SPINE = 0,
		QKINECT_UPPER_SPINE,
		QKINECT_NECK,
		QKINECT_LEFT_CLAVICLE,
		QKINECT_LEFT_UPPER_ARM,
		QKINECT_LEFT_LOWER_ARM,
		QKINECT_LEFT_METACARPAL,
		QKINECT_RIGHT_CLAVICLE,
		QKINECT_RIGHT_UPPER_ARM,
		QKINECT_RIGHT_LOWER_ARM,
		QKINECT_RIGHT_METACARPAL,
		QKINECT_LEFT_HIP,
		QKINECT_LEFT_FEMUR,
		QKINECT_LEFT_LOWER_LEG,
		QKINECT_LEFT_METATARSUS,
		QKINECT_RIGHT_HIP,
		QKINECT_RIGHT_FEMUR,
		QKINECT_RIGHT_LOWER_LEG,
		QKINECT_RIGHT_METATARSUS,
		QKINECT_NUM_BONES
	};

	virtual ~QKinect();

	int elevationAngle() const;
	// returns whether it succeeded
	bool setElevationAngle( int degrees );

	bool isNearModeEnabled() const;
	// returns whether it succeeded
	bool setNearModeEnabled( bool b );

	// set b to true to enable the emitter
	// false to disable to emitter
	// returns whether it succeeded
	bool setInfraredEmitterEnabled( bool b );

	// returns the raw accelerometer reading
	// in units of g, with y pointing down and z pointing out of the camera
	// (w = 0)
	bool rawAccelerometerReading( Vector4f& reading ) const;

	// returns the camera "up" vector using the raw accelerometer reading
	// with y pointing up
	// equal to -rawAccelerometerReading.xyz().normalized()
	Vector3f upVectorFromAccelerometer() const;

public slots:

	// poll the Kinect for data and return the results in the output variables
	// may wait up until waitIntervalMilliseconds before returning
	// setting waitIntervalMilliseconds to 0 means check and if data is ready, return immediately
	// otherwise return nothing
	//
	// returns an enumerated event indicating the result
	QKinectEvent poll( NUI_SKELETON_FRAME& skeleton, Image4ub& rgba, Array2D< ushort >& depth, Array2D< ushort >& playerIndex,
		int waitIntervalMilliseconds = 0 );

	// poll the Kinect for depth data only and return the result in the depth output variable
	// may wait up until waitIntervalMilliseconds before returning
	// setting waitIntervalMilliseconds to 0 means check and if data is ready, return immediately
	// otherwise return nothing
	//
	// on output:
	//
	// if usingExtendedDepth():
	//   each pixel of depth is set to the full extended depth range (16 bits, no shifting)
	//
	//   if usingPlayerIndex():
	//     each pixel of playerIndex is set to the player index
	// 	
	// else (packed depth):
	//   the playerIndex array is *not touched*
	//   each pixel of depth's bits 15:3 is set to its depth in millimeters
	//
	//   if usingPlayerIndex();
	//     each pixel of depth's bits 2:0 is set to the player index (0 for no player, 1-6 for tracked players)
	//   else
	//     each pixel of depth's bits 2:0 is set to 0
	// 
	QKinectEvent pollDepth( Array2D< ushort >& depth, Array2D< ushort >& playerIndex, int waitIntervalMilliseconds = 0 );

	// poll the Kinect for a voice recognition command and return the results in the output variables
	// returns false is nothing was recognized
	bool pollSpeech( QString& phrase, float& confidence, int waitInterval = 1000 );

	bool isUsingDepth() const;
	bool isUsingPlayerIndex() const;
	bool isUsingExtendedDepth() const;

private:

	QKinect( int deviceIndex );

	// initialization
	HRESULT initialize
	(
		DWORD nuiFlags,
		NUI_IMAGE_RESOLUTION rgbResolution,
		NUI_IMAGE_RESOLUTION depthResolution,
		bool usingExtendedDepth,
		QVector< QString > recognizedPhrases
	);
		HRESULT initializeDepthStream( bool trackPlayerIndex, NUI_IMAGE_RESOLUTION depthResolution, bool usingExtendedDepth );

		HRESULT initializeRGBStream();
		HRESULT initializeSkeletonTracking();
	
	HRESULT initializeSpeechRecognition( QVector< QString > recognizedPhrases );
		HRESULT initializeAudio();
		HRESULT initializePhrases( QVector< QString > recognizedPhrases );

	// convert data into recognizable formats
	bool handleGetSkeletonFrame( NUI_SKELETON_FRAME& skeleton );
	bool handleGetColorFrame( Image4ub& rgba ); // returns false on an invalid frame from USB
	bool handleGetDepthFrame( Array2D< ushort >& depth, Array2D< ushort >& playerIndex ); // returns false on an invalid frame from USB

	int m_deviceIndex;
	INuiSensor* m_pSensor;
	bool m_usingColor;
	bool m_usingDepth;
	bool m_usingPlayerIndex;
	NUI_IMAGE_RESOLUTION m_rgbResolution;
	NUI_IMAGE_RESOLUTION m_depthResolution;
	bool m_usingExtendedDepth;
	bool m_usingSkeleton;
	bool m_usingAudio;

	std::vector< HANDLE > m_eventHandles;
	HANDLE m_hNextSkeletonEvent;
	HANDLE m_hNextRGBFrameEvent;
	HANDLE m_hNextDepthFrameEvent;

	std::vector< QKinectEvent > m_eventEnums;

	HANDLE m_hRGBStreamHandle;
	HANDLE m_hDepthStreamHandle;

	// properties
	DWORD m_depthStreamFlags;

	// speech
	IMediaObject* m_pDMO;
	IPropertyStore* m_pPS;
	ISpRecoContext* m_pContext;
	KinectStream* m_pKS;
	IStream* m_pStream;
	ISpRecognizer* m_pRecognizer;
	ISpStream* m_pSpStream;
	ISpObjectToken* m_pEngineToken;
	ISpRecoGrammar* m_pGrammar;

	static std::vector< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX > > s_jointIndicesForBones;
	static std::map< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX >, int > s_boneIndicesForJoints;
};
