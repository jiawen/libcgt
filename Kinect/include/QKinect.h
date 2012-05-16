#pragma once

#include <QObject>
#include <QVector>
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

#include <memory>
#include <map>
#include <vector>

#include <common/Array2D.h>
#include <imageproc/Image4ub.h>

#include "KinectStream.h"

class QKinect : public QObject
{
	Q_OBJECT

public:

	static int numDevices();

	// Creates a Kinect device
	// deviceIndex should be between [0, numDevices()),
	// nuiFlags is an bit mask combination of NUI_INITIALIZE_FLAG_USES_COLOR, NUI_INITIALIZE_FLAG_USES_DEPTH, NUI_INITIALIZE_FLAG_USES_SKELETON, NUI_INITIALIZE_FLAG_USES_AUDIO
	// NUI_INITIALIZE_FLAG_USES_AUDIO must be set if you want to recognize any of the phrases in recognizedPhrases
	static std::shared_ptr< QKinect > create
	(
		int deviceIndex = 0,
		DWORD nuiFlags = NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH,
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

	int elevationAngle();
	void setElevationAngle( int degrees );

public slots:

	// poll the Kinect for a skeleton and return the results in the output variables
	// returns:
	//   -1 if nothing is ready (timed out)
	//   0, if skeleton frame is ready
	//   1, if rgba frame is ready
	//   2, if depth frame is ready
	QKinectEvent poll( NUI_SKELETON_FRAME& skeleton, Image4ub& rgba, Array2D< ushort >& depth, int waitInterval = 0 );

	// poll the Kinect for a voice recognition command and return the results in the output variables
	// returns false is nothing was recognized
	bool pollSpeech( QString& phrase, float& confidence, int waitInterval = 1000 );

private:

	QKinect( int deviceIndex );

	// initialization
	HRESULT initialize( DWORD nuiFlags, QVector< QString > recognizedPhrases );
		HRESULT initializeDepthStream();
		HRESULT initializeRGBStream();
		HRESULT initializeSkeletonTracking();
	
	HRESULT initializeSpeechRecognition( QVector< QString > recognizedPhrases );
		HRESULT initializeAudio();
		HRESULT initializePhrases( QVector< QString > recognizedPhrases );

	// convert data into recognizable formats
	bool handleGetSkeletonFrame( NUI_SKELETON_FRAME& skeleton );
	bool handleGetColorFrame( Image4ub& rgba ); // returns false on an invalid frame from USB
	bool handleGetDepthFrame( Array2D< ushort >& depth ); // returns false on an invalid frame from USB

	int m_deviceIndex;
	INuiSensor* m_pSensor;
	
	std::vector< HANDLE > m_eventHandles;
	HANDLE m_hNextSkeletonEvent;
	HANDLE m_hNextRGBFrameEvent;
	HANDLE m_hNextDepthFrameEvent;

	std::vector< QKinectEvent > m_eventEnums;

	HANDLE m_hRGBStreamHandle;
	HANDLE m_hDepthStreamHandle;

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
