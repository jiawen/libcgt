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

#include <cstdint>
#include <map>
#include <vector>

#include <common/Array2D.h>
#include <common/BasicTypes.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

#include "KinectStream.h"

// TODO: Switch enums into enum classes.
// TODO: Consider getting rid of a copy.
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
        NUI_IMAGE_RESOLUTION colorResolution = NUI_IMAGE_RESOLUTION_640x480,
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

    enum class Event
    {
        FAILED,
        TIMEOUT,
        SKELETON,
        COLOR,
        DEPTH
    };

    enum class Bone
    {
        LOWER_SPINE = 0,
        UPPER_SPINE,
        NECK,
        LEFT_CLAVICLE,
        LEFT_UPPER_ARM,
        LEFT_LOWER_ARM,
        LEFT_METACARPAL,
        RIGHT_CLAVICLE,
        RIGHT_UPPER_ARM,
        RIGHT_LOWER_ARM,
        RIGHT_METACARPAL,
        LEFT_HIP,
        LEFT_FEMUR,
        LEFT_LOWER_LEG,
        LEFT_METATARSUS,
        RIGHT_HIP,
        RIGHT_FEMUR,
        RIGHT_LOWER_LEG,
        RIGHT_METATARSUS,
        NUM_BONES
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

    // Poll the Kinect for data and writes the results into one of the output
    // variables, returning which output variable (if any) was written.
    // It waits up to waitIntervalMilliseconds for data to be
    // available. If waitIntervalMilliseconds is 0, it will return immediately,
    // with the data if it's available. Else, timeout.
    QKinect::Event poll( NUI_SKELETON_FRAME& skeleton,
        Array2DView< uint8x4 > bgra,
        Array2DView< uint16_t > depth,
        Array2DView< uint16_t > playerIndex,
        int waitIntervalMilliseconds = 0 );

    // Poll the Kinect for color data and if data is available, write it to
    // rgba. It waits up to waitIntervalMilliseconds for data to be
    // available. If waitIntervalMilliseconds is 0, it will return immediately,
    // with the data if it's available. Else, timeout.
    QKinect::Event QKinect::pollColor( Array2DView< uint8x4 > bgra,
        int waitIntervalMilliseconds = 0 );

    // Poll the Kinect for depth data and if data is available, write it to
    // rgba. It waits up to waitIntervalMilliseconds for data to be
    // available. If waitIntervalMilliseconds is 0, it will return immediately,
    // with the data if it's available. Else, timeout.
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
    QKinect::Event pollDepth( Array2DView< uint16_t > depth, Array2DView< uint16_t > playerIndex, int waitIntervalMilliseconds = 0 );

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
        NUI_IMAGE_RESOLUTION colorResolution,
        NUI_IMAGE_RESOLUTION depthResolution,
        bool usingExtendedDepth,
        QVector< QString > recognizedPhrases
    );
        HRESULT initializeColorStream( NUI_IMAGE_RESOLUTION resolution );
        HRESULT initializeDepthStream( bool trackPlayerIndex,
            NUI_IMAGE_RESOLUTION resolution, bool usingExtendedDepth );
        HRESULT initializeSkeletonTracking();

    HRESULT initializeSpeechRecognition( QVector< QString > recognizedPhrases );
        HRESULT initializeAudio();
        HRESULT initializePhrases( QVector< QString > recognizedPhrases );

    // Convert data into recognizable formats.
    bool handleGetSkeletonFrame( NUI_SKELETON_FRAME& skeleton );

    // Returns false on an invalid frame from USB.
    bool handleGetColorFrame( Array2DView< uint8x4 > bgra );

    // Returns false on an invalid frame from USB.
    bool handleGetDepthFrame( Array2DView< uint16_t > depth,
        Array2DView< uint16_t > playerIndex );

    int m_deviceIndex;
    INuiSensor* m_pSensor;
    bool m_usingColor;
    bool m_usingDepth;
    bool m_usingPlayerIndex;
    NUI_IMAGE_RESOLUTION m_colorResolution;
    Vector2i m_colorResolutionPixels;
    NUI_IMAGE_RESOLUTION m_depthResolution;
    Vector2i m_depthResolutionPixels;
    bool m_usingExtendedDepth;
    bool m_usingSkeleton;
    bool m_usingAudio;

    std::vector< HANDLE > m_eventHandles;
    HANDLE m_hNextSkeletonEvent;
    HANDLE m_hNextColorFrameEvent;
    HANDLE m_hNextDepthFrameEvent;

    std::vector< QKinect::Event > m_eventEnums;

    HANDLE m_hColorStreamHandle;
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
