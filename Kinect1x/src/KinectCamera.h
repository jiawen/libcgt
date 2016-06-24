#pragma once

#define KINECT1X_ENABLE_SPEECH 0

#if KINECT1X_ENABLE_SPEECH
// In Visual Studio 2010
#if ( _MSC_VER <= 1600 )
#pragma warning( disable: 4005 ) // stdint.h and intsafe.h: disable warnings for multiply defined.
#pragma warning( disable: 4805 ) // sphelper.h: disable unsafe mix of BOOL and bool.
#endif
#pragma warning( disable: 4996 ) // sphelper.h: disable GetVersionExW declared deprecated.
#endif

#include <windows.h>
#include <NuiApi.h>

#if KINECT1X_ENABLE_SPEECH
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
#endif

#include <cstdint>
#include <map>
#include <vector>

#include <cameras/Intrinsics.h>
#include <common/Array2D.h>
#include <common/BasicTypes.h>
#include <vecmath/EuclideanTransform.h>
#include <vecmath/Range1f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

#if KINECT1X_ENABLE_SPEECH
#include "KinectStream.h"
#endif

namespace libcgt { namespace kinect1x {

// TODO: Replace pointer with RAII.
// TODO: expose infrared stream
class KinectCamera
{
    using Intrinsics = libcgt::core::cameras::Intrinsics;
    using EuclideanTransform = libcgt::core::vecmath::EuclideanTransform;

public:

    static int numDevices();

    // 800 mm.
    static uint16_t minimumDepthMillimeters();

    // 4000 mm.
    static uint16_t maximumDepthMillimeters();

    // 0.8m - 4m.
    static Range1f depthRangeMeters();

    // 400 mm.
    static uint16_t nearModeMinimumDepthMillimeters();

    // 3000 mm.
    static uint16_t nearModeMaximumDepthMillimeters();

    // 0.4m - 3m.
    static Range1f nearModeDepthRangeMeters();

    // The typical factory-calibrated intrinsics of the Kinect color camera.
    // Returns {{0, 0}, {0, 0}} for an invalid resolution.
    static Intrinsics colorIntrinsics( NUI_IMAGE_RESOLUTION resolution );

    // The typical factory-calibrated intrinsics of the Kinect depth camera.
    // Returns {{0, 0}, {0, 0}} for an invalid resolution.
    static Intrinsics depthIntrinsics( NUI_IMAGE_RESOLUTION resolution );

    // Get the transformation mapping: color_coord <-- depth_coord.
    // This transformation is in the OpenGL convention (y-up, z-out-of-screen).
    // Translation has units of millimeters.
    static EuclideanTransform colorFromDepthExtrinsicsMillimeters();

    // Get the transformation mapping: color_coord <-- depth_coord.
    // This transformation is in the OpenGL convention (y-up, z-out-of-screen).
    // Translation has units of millimeters.
    static EuclideanTransform colorFromDepthExtrinsicsMeters();

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
    static KinectCamera* create
    (
        int deviceIndex = 0,
        DWORD nuiFlags = NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH,
        NUI_IMAGE_RESOLUTION colorResolution = NUI_IMAGE_RESOLUTION_640x480,
        NUI_IMAGE_RESOLUTION depthResolution = NUI_IMAGE_RESOLUTION_640x480
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
        DEPTH,
        RGBD
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

    virtual ~KinectCamera();

    // Get the current elevation angle in degrees.
    int elevationAngle() const;

    // Set the camera elevation angle. Returns whether it succeeded.
    bool setElevationAngle( int degrees );

    // Returns true if the depth stream is currently has near mode enabled.
    bool isNearModeEnabled() const;

    // Set the depth stream in near mode. Returns whether it succeeded.
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

    // The typical factory-calibrated intrinsics of the Kinect color camera.
    // Returns {{0, 0}, {0, 0}} if not configured.
    libcgt::core::cameras::Intrinsics colorIntrinsics() const;

    // The typical factory-calibrated intrinsics of the Kinect depth camera.
    // Returns {{0, 0}, {0, 0}} if not configured.
    libcgt::core::cameras::Intrinsics depthIntrinsics() const;

    // Block until one of the input channels is available. This function blocks
    // for up to waitIntervalMilliseconds for data to be available. If
    // waitIntervalMilliseconds is 0, it will return immediately (whether data
    // is available or not).
    //
    // The return value indicates which channel was written to.
    Event poll( NUI_SKELETON_FRAME& skeleton,
        Array2DView< uint8x4 > bgra,
        int64_t& bgraTimestamp, int& bgraFrameNumber,
        Array2DView< uint16_t > depth,
        Array2DView< uint16_t > playerIndex,
        int64_t& depthTimestamp, int& depthFrameNumber,
        bool& depthCapturedWithNearMode,
        int waitIntervalMilliseconds = INFINITE );

    // Block until color, extended depth, and optionally player index are all
    // simultaneously available. Pass a view that isNull() to ignore that
    // stream.
    Event pollColorAndExtendedDepth( Array2DView< uint8x4 > bgra,
        int64_t& bgraTimestamp, int& bgraFrameNumber,
        Array2DView< uint16_t > depth,
        Array2DView< uint16_t > playerIndex,
        int64_t& depthTimestamp, int& depthFrameNumber,
        bool& depthCapturedWithNearMode,
        int waitIntervalMilliseconds = INFINITE );

    // Block until color data is available. This function blocks for up to
    // waitIntervalMilliseconds for data to be available. If
    // waitIntervalMilliseconds is 0, it will return immediately (whether data
    // is available or not).
    Event pollColor( Array2DView< uint8x4 > bgra,
        int64_t& timestamp, int& frameNumber,
        int waitIntervalMilliseconds = INFINITE );

    // Block until depth data is available. This function blocks for up to
    // waitIntervalMilliseconds for data to be available. If
    // waitIntervalMilliseconds is 0, it will return immediately (whether data
    // is available or not).
    //
    // Each pixel of depth is set to the full sensor depth range in millimeters
    // using the full 16-bit range (no bit shifting). If skeletal or player
    // tracking is enabled, then each pixel is set to the player index (1-6),
    // or 0 for no player (and ignored if tracking is disabled).
    //
    // Pass a view that isNull() to ignore that stream.
    Event pollExtendedDepth( Array2DView< uint16_t > depth,
        Array2DView< uint16_t > playerIndex,
        int64_t& timestamp, int& frameNumber, bool& capturedWithNearMode,
        int waitIntervalMilliseconds = INFINITE );

    // Block until depth data is available. This function blocks for up to
    // waitIntervalMilliseconds for data to be available. If
    // waitIntervalMilliseconds is 0, it will return immediately (whether data
    // is available or not).
    //
    // Each pixel of packedDepth is populated with a 16-bit value where:
    // - bits 15:3 is depth in millimeters.
    // - If skeletal and player tracking is enabled, then bits 2:0 are set to
    //   the player index (1-6), or 0 for no player. If tracking is disabled,
    //   then the bits 2:0 are set to 0.
    //
    // Pass a view that isNull() to ignore that stream.
    Event pollPackedDepth( Array2DView< uint16_t > packedDepth,
        int waitIntervalMilliseconds = INFINITE );

#if KINECT1X_ENABLE_SPEECH
    // poll the Kinect for a voice recognition command and return the results in the output variables
    // returns false is nothing was recognized
    bool pollSpeech( QString& phrase, float& confidence,
        int waitInterval = 1000 );
#endif

    // Which streams are enabled.

    bool isUsingColor() const;

    bool isUsingDepth() const;

    bool isUsingPlayerIndex() const;

    bool isUsingSkeleton() const;

private:

    KinectCamera( int deviceIndex );

    // initialization
    HRESULT initialize( DWORD nuiFlags, NUI_IMAGE_RESOLUTION colorResolution,
        NUI_IMAGE_RESOLUTION depthResolution );
        HRESULT initializeColorStream( NUI_IMAGE_RESOLUTION resolution );
        HRESULT initializeDepthStream( NUI_IMAGE_RESOLUTION resolution,
            bool trackPlayerIndex );
        HRESULT initializeSkeletonTracking();

#if KINECT1X_ENABLE_SPEECH
    HRESULT initializeSpeechRecognition( QVector< QString > recognizedPhrases );
        HRESULT initializeAudio();
        HRESULT initializePhrases( QVector< QString > recognizedPhrases );
#endif

    // Convert data into recognizable formats.
    bool handleGetSkeletonFrame( NUI_SKELETON_FRAME& skeleton );

    // Returns false on an invalid frame from USB.
    bool handleGetColorFrame( Array2DView< uint8x4 > bgra,
        int64_t& timestamp, int& frameNumber );

    // Returns false on an invalid frame from USB.
    bool handleGetExtendedDepthFrame( Array2DView< uint16_t > depth,
        Array2DView< uint16_t > playerIndex,
        int64_t& timestamp, int& frameNumber, bool& capturedWithNearMode );

    // Returns false on an invalid frame from USB.
    bool handleGetPackedDepthFrame( Array2DView< uint16_t > depth );

    int m_deviceIndex = -1;
    INuiSensor* m_pSensor = nullptr;
    bool m_usingColor = false;
    bool m_usingDepth = false;
    bool m_usingPlayerIndex = false;
    NUI_IMAGE_RESOLUTION m_colorResolution = NUI_IMAGE_RESOLUTION_INVALID;
    Vector2i m_colorResolutionPixels;
    NUI_IMAGE_RESOLUTION m_depthResolution = NUI_IMAGE_RESOLUTION_INVALID;
    Vector2i m_depthResolutionPixels;
    bool m_usingSkeleton = false;
    bool m_usingAudio = false;

    HANDLE m_hNextSkeletonEvent = NULL;
    HANDLE m_hNextColorFrameEvent = NULL;
    HANDLE m_hNextDepthFrameEvent = NULL;

    HANDLE m_hColorStreamHandle = NULL;
    HANDLE m_hDepthStreamHandle = NULL;

    // Current flags for the depth stream.
    DWORD m_depthStreamFlags = 0;

#if KINECT1X_ENABLE_SPEECH
    // speech
    IMediaObject* m_pDMO = nullptr;
    IPropertyStore* m_pPS = nullptr;
    ISpRecoContext* m_pContext = nullptr;
    KinectStream* m_pKS = nullptr;
    IStream* m_pStream = nullptr;
    ISpRecognizer* m_pRecognizer = nullptr;
    ISpStream* m_pSpStream = nullptr;
    ISpObjectToken* m_pEngineToken = nullptr;
    ISpRecoGrammar* m_pGrammar = nullptr;
#endif

    static std::vector< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX > > s_jointIndicesForBones;
    static std::map< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX >, int > s_boneIndicesForJoints;
};

} } // kinect1x, libcgt
