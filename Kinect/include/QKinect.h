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
	static std::shared_ptr< QKinect > create( int deviceIndex = 0, QVector< QString > recognizedPhrases = QVector< QString >() );

	// returns a vector of pairs of indices (i,j)
	// such that within a NUI_SKELETON_FRAME
	// frame.SkeletonData[k].SkeletonPositions[i] --> frame.SkeletonData[k].SkeletonPositions[j] is a bone
	static const std::vector< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX > >& jointIndicesForBones();

	// returns the inverse of jointIndicesForBones
	static const std::map< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX >, int >& boneIndicesForJoints();

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
	//   0, if skeleton frame is ready
	//   1, if rgba frame is ready
	//   -1, if failed or timeout
	int poll( NUI_SKELETON_FRAME& skeletonFrame, Image4ub& rgba, Array2D< ushort >& depth, int waitInterval = 0 );

	bool pollSpeech( QString& phrase, float& confidence, int waitInterval = 1000 );

signals:

private:

	QKinect();

	static HRESULT initializeSkeletonTracking( QKinect* pKinect );
	
	static HRESULT initializeSpeechRecognition( QKinect* pKinect, QVector< QString > recognizedPhrases );
		static HRESULT initializeAudio( QKinect* pKinect );
		static HRESULT initializePhrases( QKinect* pKinect, QVector< QString > recognizedPhrases );

	NUI_SKELETON_FRAME handleGetSkeletonFrame();
	bool handleGetColorFrame( Image4ub& rgba ); // returns false on an invalid frame from USB
	bool handleGetDepthFrame( Array2D< ushort >& depth ); // returns false on an invalid frame from USB

	INuiSensor* m_pSensor;
	
	HANDLE m_events[3];
	HANDLE m_hNextSkeletonEvent;
	HANDLE m_hNextVideoFrameEvent;
	HANDLE m_hNextDepthFrameEvent;

	HANDLE m_hVideoStreamHandle;
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
