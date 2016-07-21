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

class KinectCamera
{
public:

    using Intrinsics = libcgt::core::cameras::Intrinsics;
    using EuclideanTransform = libcgt::core::vecmath::EuclideanTransform;

    enum class Event
    {
        FAILED,
        TIMEOUT,
        OK
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

    // A simple (non-owning) container for a frame of data and its metadata
    // returned by the camera. When polling from the camera, "colorUpdated",
    // "depthUpdated", and/or "skeletonUpdated" will be set to true indicating
    // which was updated.
    struct Frame
    {
        // ----- Color stream -----
        // Only one of bgra or infrared will be populated, depending on which
        // color format was configured.
        bool colorUpdated;

        int64_t colorTimestamp;
        int colorFrameNumber;
        // bgra will always have alpha set to 0.
        Array2DView< uint8x4 > bgra;
        // The infrared stream has data only in the top 10 bits. Bits 5:0 are 0.
        Array2DView< uint16_t > infrared;
        // TODO(jiawen): YUV formats.

        // ----- Depth stream -----
        // Only one of packedDepth or (extendedDepth and playerIndex) will be
        // populated, depending on whether player tracking is enabled, and
        // whether the client requested packed or extended depth.
        bool depthUpdated;

        int64_t depthTimestamp;
        int depthFrameNumber;
        bool depthCapturedWithNearMode; // Not updated if using packed depth.

        // Each pixel of packedDepth is populated with a 16-bit value where:
        // - bits 15:3 is depth in millimeters.
        // - If player tracking is enabled, then bits 2:0 are set to the player
        //   index (1-6), or 0 for no player. If tracking is disabled, then
        //   bits 2:0 are set to 0.
        Array2DView< uint16_t > packedDepth;

        // Each pixel of extendedDepth is set to the full sensor depth range in
        // millimeters using the full 16-bit range (no bit shifting).
        Array2DView< uint16_t > extendedDepth;
        // If player tracking is enabled, then each pixel is set to the player
        // index (1-6), or 0 for no player.
        Array2DView< uint16_t > playerIndex;

        bool skeletonUpdated;
        NUI_SKELETON_FRAME skeleton;
    };

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

    // Returns a vector of pairs of indices (i,j) such that within a
    // NUI_SKELETON_FRAME k, frame.SkeletonData[k].SkeletonPositions[i]
    // --> frame.SkeletonData[k].SkeletonPositions[j] is a bone.
    static const std::vector< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX > >& jointIndicesForBones();

    // The inverse of jointIndicesForBones().
    static const std::map< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX >, int >& boneIndicesForJoints();

    // Create a Kinect device.
    //
    // deviceIndex should be between [0, numDevices()),
    //
    // nuiFlags is an bit mask combination of:
    //   NUI_INITIALIZE_FLAG_USES_COLOR,
    //   NUI_INITIALIZE_FLAG_USES_DEPTH,
    //   NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX,
    //   NUI_INITIALIZE_FLAG_USES_SKELETON,
    //   NUI_INITIALIZE_FLAG_USES_AUDIO
    //
    //   Restrictions:
    //   - To enable skeleton tracking (NUI_INITIALIZE_FLAG_USES_SKELETON set),
    //     NUI_INITIALIZE_FLAG_USES_DEPTH or
    //     NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX must be set.
    //   Notes:
    //   - INFRARED images are imaged with the depth sensor, but arrive on the
    //     COLOR stream.
    //
    // colorResolution and depthResolution are self-explanatory.
    //
    ///  Restrictions:
    //   - colorResolution depends on the colorType:
    //     - NUI_IMAGE_TYPE_COLOR: 640x480 or 1280x960.
    //     - NUI_IMAGE_TYPE_COLOR_YUV: 640x480
    //     - NUI_IMAGE_TYPE_COLOR_RAW_YUV: 640x480
    //     - NUI_IMAGE_TYPE_COLOR_INFRARED: 640x480
    //   - depthResolution depends on whether player index tracking is enabled.
    //     - DEPTH: 80x60, 320x240, or 640x480.
    //     - DEPTH_AND_PLAYER_INDEX: 80,60 or 320x240.
    KinectCamera( int deviceIndex = 0,
        DWORD nuiFlags =
            NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH,
        NUI_IMAGE_TYPE colorFormat = NUI_IMAGE_TYPE_COLOR,
        NUI_IMAGE_RESOLUTION colorResolution = NUI_IMAGE_RESOLUTION_640x480,
        NUI_IMAGE_RESOLUTION depthResolution = NUI_IMAGE_RESOLUTION_640x480 );

    KinectCamera( const KinectCamera& copy ) = delete;
    KinectCamera& operator = ( const KinectCamera& copy ) = delete;
    // TODO(VS2015): move constructor = default
    virtual ~KinectCamera();

    // Returns true if the camera is correctly initialized.
    bool isValid() const;

    // Get the current elevation angle in degrees.
    int elevationAngle() const;

    // Set the camera elevation angle. Returns whether it succeeded.
    bool setElevationAngle( int degrees );

    // Returns true if the depth stream is currently has near mode enabled.
    bool isNearModeEnabled() const;

    // Set the depth stream in near mode. Returns whether it succeeded.
    bool setNearModeEnabled( bool b );

    // Set b to true to enable the emitter,  false to disable to emitter.
    // Returns whether it succeeded.
    //
    // Note that this only works on the Kinect for Windows sensor (and not the
    // Kinect for Xbox 360 sensor).
    bool setInfraredEmitterEnabled( bool b );

    // returns the raw accelerometer reading
    // in units of g, with y pointing down and z pointing out of the camera
    // (w = 0)
    bool rawAccelerometerReading( Vector4f& reading ) const;

    // returns the camera "up" vector using the raw accelerometer reading
    // with y pointing up
    // equal to -rawAccelerometerReading.xyz().normalized()
    Vector3f upVectorFromAccelerometer() const;

    // The resolution of the configured color or infrared stream.
    // Returns (0, 0) if it's not configured.
    Vector2i colorResolution() const;

    // The typical factory-calibrated intrinsics of the Kinect color camera.
    // Returns {{0, 0}, {0, 0}} if not configured.
    Intrinsics colorIntrinsics() const;

    // The resolution of the configured depth stream.
    // Returns (0, 0) if it's not configured.
    Vector2i depthResolution() const;

    // The typical factory-calibrated intrinsics of the Kinect depth camera.
    // Returns {{0, 0}, {0, 0}} if not configured.
    Intrinsics depthIntrinsics() const;

    // Block until one of the input streams is available. This function blocks
    // for up to waitIntervalMilliseconds for data to be available. If
    // waitIntervalMilliseconds is 0, it will return immediately (whether data
    // is available or not).
    //
    // The "updated" flag in "frame" corresponding to the channel will be set.
    Event pollOne( Frame& frame,
        bool useExtendedDepth = true,
        int waitIntervalMilliseconds = INFINITE );

    // Block until all the input streams are available. This function blocks
    // for up to waitIntervalMilliseconds for data to be available. If
    // waitIntervalMilliseconds is 0, it will return immediately (whether data
    // is available or not).
    //
    // The "updated" flag in "frame" corresponding to the channels will be set.
    Event pollAll( Frame& frame,
        bool useExtendedDepth = true,
        int waitIntervalMilliseconds = INFINITE );

#if KINECT1X_ENABLE_SPEECH
    // poll the Kinect for a voice recognition command and return the results in the output variables
    // returns false is nothing was recognized
    bool pollSpeech( QString& phrase, float& confidence,
        int waitInterval = 1000 );
#endif

    // Which streams are enabled.

    bool isUsingColor() const;
    NUI_IMAGE_TYPE colorFormat() const;

    bool isUsingDepth() const;

    bool isUsingPlayerIndex() const;

    bool isUsingSkeleton() const;

private:

    HRESULT initializeColorStream( NUI_IMAGE_TYPE format,
        NUI_IMAGE_RESOLUTION resolution );
    HRESULT initializeDepthStream( NUI_IMAGE_RESOLUTION resolution,
        bool trackPlayerIndex );
    HRESULT initializeSkeletonTracking();

    void close();

#if KINECT1X_ENABLE_SPEECH
    HRESULT initializeSpeechRecognition( QVector< QString > recognizedPhrases );
        HRESULT initializeAudio();
        HRESULT initializePhrases( QVector< QString > recognizedPhrases );
#endif

    // Helper function for pollOne() and pollAll() to populate "frame" once
    // event "eventIndex" has been signaled. Returns whether it was successful.
    bool handleEvent( Frame& frame, bool useExtendedDepth, DWORD eventIndex );

    // Convert data into recognizable formats.
    bool handleGetSkeletonFrame( NUI_SKELETON_FRAME& skeleton );

    // Returns false if the input is invalid (incorrect size or format), or
    // on an invalid frame from the sensor.
    bool handleGetColorFrame( Array2DView< uint8x4 > bgra,
        int64_t& timestamp, int& frameNumber );

    // Returns false if the input is invalid (incorrect size or format), or
    // on an invalid frame from the sensor.
    bool handleGetInfraredFrame( Array2DView< uint16_t > ir,
        int64_t& timestamp, int& frameNumber );

    // Returns false if the input is invalid (incorrect size or format), or
    // on an invalid frame from the sensor.
    bool handleGetExtendedDepthFrame( Array2DView< uint16_t > depth,
        Array2DView< uint16_t > playerIndex,
        int64_t& timestamp, int& frameNumber, bool& capturedWithNearMode );

    // Returns false if the input is invalid (incorrect size or format), or
    // on an invalid frame from the sensor.
    bool handleGetPackedDepthFrame( Array2DView< uint16_t > depth,
        int64_t& timestamp, int& frameNumber );

    int m_deviceIndex = -1;
    INuiSensor* m_pSensor = nullptr;
    bool m_usingPlayerIndex = false;
    NUI_IMAGE_RESOLUTION m_colorResolution = NUI_IMAGE_RESOLUTION_INVALID;
    NUI_IMAGE_TYPE m_colorFormat;
    Vector2i m_colorResolutionPixels;
    NUI_IMAGE_RESOLUTION m_depthResolution = NUI_IMAGE_RESOLUTION_INVALID;
    Vector2i m_depthResolutionPixels;
#if KINECT1X_ENABLE_SPEECH
    bool m_usingAudio = false;
#endif

    HANDLE m_hNextColorFrameEvent = NULL;
    HANDLE m_hNextDepthFrameEvent = NULL;
    HANDLE m_hNextSkeletonEvent = NULL;
    std::vector< HANDLE > m_events;

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
