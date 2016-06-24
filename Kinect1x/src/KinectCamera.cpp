#include "KinectCamera.h"

#if KINECT1X_ENABLE_SPEECH
// For string IO and manipulation.
#include <strsafe.h>
#include <conio.h>
#endif

#include <common/ArrayUtils.h>
#include <imageproc/Swizzle.h>

#include "KinectUtils.h"

using libcgt::core::arrayutils::componentView;
using libcgt::core::arrayutils::copy;
using libcgt::core::cameras::Intrinsics;
using libcgt::core::vecmath::EuclideanTransform;

namespace libcgt { namespace kinect1x {

// static
std::vector< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX > > KinectCamera::s_jointIndicesForBones;
// static
std::map< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX >, int > KinectCamera::s_boneIndicesForJoints;

// static
int KinectCamera::numDevices()
{
    int nDevices;
    NuiGetSensorCount( &nDevices );

    return nDevices;
}

// static
uint16_t KinectCamera::minimumDepthMillimeters()
{
    return NUI_IMAGE_DEPTH_MINIMUM >> NUI_IMAGE_PLAYER_INDEX_SHIFT;
}

// static
uint16_t KinectCamera::maximumDepthMillimeters()
{
    return NUI_IMAGE_DEPTH_MAXIMUM >> NUI_IMAGE_PLAYER_INDEX_SHIFT;
}

// static
Range1f KinectCamera::depthRangeMeters()
{
    return Range1f::fromMinMax
    (
        minimumDepthMillimeters() * 0.001f,
        maximumDepthMillimeters() * 0.001f
    );
}

// static
uint16_t KinectCamera::nearModeMinimumDepthMillimeters()
{
    return NUI_IMAGE_DEPTH_MINIMUM_NEAR_MODE >> NUI_IMAGE_PLAYER_INDEX_SHIFT;
}

// static
uint16_t KinectCamera::nearModeMaximumDepthMillimeters()
{
    return NUI_IMAGE_DEPTH_MAXIMUM_NEAR_MODE >> NUI_IMAGE_PLAYER_INDEX_SHIFT;
}

// static
Range1f KinectCamera::nearModeDepthRangeMeters()
{
    return Range1f::fromMinMax
    (
        nearModeMinimumDepthMillimeters() * 0.001f,
        nearModeMaximumDepthMillimeters() * 0.001f
    );
}

// static
Intrinsics KinectCamera::colorIntrinsics(
    NUI_IMAGE_RESOLUTION resolution )
{
    // #define is on a 640x480 image.
    Vector2f fl{ NUI_CAMERA_COLOR_NOMINAL_FOCAL_LENGTH_IN_PIXELS };
    Vector2f pp{ 320, 240 };

    switch( resolution )
    {
    case NUI_IMAGE_RESOLUTION_80x60:
        fl *= 0.125f;
        pp *= 0.125f;
        break;
    case NUI_IMAGE_RESOLUTION_320x240:
        fl *= 0.5f;
        pp *= 0.5f;
        break;
    case NUI_IMAGE_RESOLUTION_640x480:
        break;
    case NUI_IMAGE_RESOLUTION_1280x960:
        fl *= 2;
        pp *= 2;
        break;
    default:
        fl = Vector2f{ 0, 0 };
        pp = Vector2f{ 0.0f, 0.0f };
        break;
    }

    return Intrinsics{ fl, pp };
}

// static
Intrinsics KinectCamera::depthIntrinsics(
    NUI_IMAGE_RESOLUTION resolution )
{
    // #define is on a 320x240 image.
    Vector2f fl{ NUI_CAMERA_DEPTH_NOMINAL_FOCAL_LENGTH_IN_PIXELS };
    Vector2f pp{ 160, 120 };

    switch( resolution )
    {
    case NUI_IMAGE_RESOLUTION_320x240:
        break;
    case NUI_IMAGE_RESOLUTION_640x480:
        fl *= 2;
        pp *= 2;
        break;
    default:
        fl = Vector2f{ 0, 0 };
        pp = Vector2f{ 0.0f, 0.0f };
        break;
    }

    return Intrinsics{ fl, pp };
}

// static
EuclideanTransform KinectCamera::colorFromDepthExtrinsicsMillimeters()
{
    return
    {
        Matrix3f::identity(),
        { -25.4f, -0.13f, -2.18f }
    };
}

// static
EuclideanTransform KinectCamera::colorFromDepthExtrinsicsMeters()
{
    auto output = colorFromDepthExtrinsicsMillimeters();
    output.translation *= 0.001f;
    return output;
}

// static
KinectCamera* KinectCamera::create( int deviceIndex, DWORD nuiFlags,
    NUI_IMAGE_RESOLUTION colorResolution,
    NUI_IMAGE_RESOLUTION depthResolution )
{
    KinectCamera* pKinect = new KinectCamera( deviceIndex );

    HRESULT hr = pKinect->initialize( nuiFlags,
        colorResolution, depthResolution );
    if( SUCCEEDED( hr ) )
    {
        return pKinect;
    }
    else
    {
        delete pKinect;
        pKinect = nullptr;
    }

    return pKinect;
}

// static
const std::vector< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX > >& KinectCamera::jointIndicesForBones()
{
    if( KinectCamera::s_jointIndicesForBones.size() == 0 )
    {
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_SPINE ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_SPINE, NUI_SKELETON_POSITION_SHOULDER_CENTER ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_HEAD ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SHOULDER_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_SHOULDER_LEFT, NUI_SKELETON_POSITION_ELBOW_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_ELBOW_LEFT, NUI_SKELETON_POSITION_WRIST_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_WRIST_LEFT, NUI_SKELETON_POSITION_HAND_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SHOULDER_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_SHOULDER_RIGHT, NUI_SKELETON_POSITION_ELBOW_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_ELBOW_RIGHT, NUI_SKELETON_POSITION_WRIST_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_WRIST_RIGHT, NUI_SKELETON_POSITION_HAND_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_HIP_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_HIP_LEFT, NUI_SKELETON_POSITION_KNEE_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_KNEE_LEFT, NUI_SKELETON_POSITION_ANKLE_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_ANKLE_LEFT, NUI_SKELETON_POSITION_FOOT_LEFT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_HIP_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_HIP_RIGHT, NUI_SKELETON_POSITION_KNEE_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_KNEE_RIGHT, NUI_SKELETON_POSITION_ANKLE_RIGHT ) );
        s_jointIndicesForBones.push_back( std::make_pair( NUI_SKELETON_POSITION_ANKLE_RIGHT, NUI_SKELETON_POSITION_FOOT_RIGHT ) );
    }

    return s_jointIndicesForBones;
}

// static
const std::map< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX >, int >& KinectCamera::boneIndicesForJoints()
{
    if( s_boneIndicesForJoints.size() == 0 )
    {
        const auto& jifb = jointIndicesForBones();
        for( int b = 0; b < jifb.size(); ++b )
        {
            auto j0j1 = jifb[b];
            s_boneIndicesForJoints[ j0j1 ] = b;
        }
    }

    return s_boneIndicesForJoints;
}

// virtual
KinectCamera::~KinectCamera()
{
#if KINECT1X_ENABLE_SPEECH
    if( m_pKS != nullptr )
    {
        m_pKS->StopCapture();
    }

    SAFE_RELEASE( m_pKS );
    SAFE_RELEASE( m_pStream );
    SAFE_RELEASE( m_pRecognizer );
    SAFE_RELEASE( m_pSpStream );
    SAFE_RELEASE( m_pEngineToken );
    SAFE_RELEASE( m_pContext );
    SAFE_RELEASE( m_pGrammar );
    SAFE_RELEASE( m_pPS );
    SAFE_RELEASE( m_pDMO );
#endif

    // no need to delete the streams, they are released per frame
    if( m_hNextDepthFrameEvent != NULL &&
        m_hNextDepthFrameEvent != INVALID_HANDLE_VALUE )
    {
        CloseHandle( m_hNextDepthFrameEvent );
        m_hNextDepthFrameEvent = NULL;
    }

    if( m_hNextColorFrameEvent != NULL &&
        m_hNextColorFrameEvent != INVALID_HANDLE_VALUE )
    {
        CloseHandle( m_hNextColorFrameEvent );
        m_hNextColorFrameEvent = NULL;
    }

    if( m_hNextSkeletonEvent != NULL &&
        m_hNextSkeletonEvent != INVALID_HANDLE_VALUE )
    {
        CloseHandle( m_hNextSkeletonEvent );
        m_hNextSkeletonEvent = NULL;
    }

    if( m_pSensor != nullptr )
    {
        m_pSensor->NuiShutdown();
    }
}

int KinectCamera::elevationAngle() const
{
    LONG degrees;
    m_pSensor->NuiCameraElevationGetAngle( &degrees );
    return static_cast< int >( degrees );
}

bool KinectCamera::setElevationAngle( int degrees )
{
    HRESULT hr = m_pSensor->NuiCameraElevationSetAngle( degrees );
    return SUCCEEDED( hr );
}

bool KinectCamera::isNearModeEnabled() const
{
    return( ( m_depthStreamFlags & NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE ) != 0 );
}

bool KinectCamera::setNearModeEnabled( bool b )
{
    // Only do something if the new state is different.
    if( b ^ isNearModeEnabled() )
    {
        // NuiImageStreamSetImageFrameFlags will fail with E_INVALID_ARG if
        // depth was not enabled.
        DWORD flags =
            m_depthStreamFlags | NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE;
        HRESULT hr = NuiImageStreamSetImageFrameFlags(
            m_hDepthStreamHandle, m_depthStreamFlags );
        if( SUCCEEDED( hr ) )
        {
            m_depthStreamFlags = flags;
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        return true;
    }
}

bool KinectCamera::setInfraredEmitterEnabled( bool b )
{
    HRESULT hr = m_pSensor->NuiSetForceInfraredEmitterOff( b );
    return SUCCEEDED( hr );
}

bool KinectCamera::rawAccelerometerReading( Vector4f& reading ) const
{
    Vector4* pReading = reinterpret_cast<Vector4*>( &reading );
    HRESULT hr = m_pSensor->NuiAccelerometerGetCurrentReading( pReading );
    return SUCCEEDED( hr );
}

Vector3f KinectCamera::upVectorFromAccelerometer() const
{
    Vector4f reading;
    Vector4* pReading = reinterpret_cast< Vector4* >( &reading );
    HRESULT hr = m_pSensor->NuiAccelerometerGetCurrentReading( pReading );
    return -reading.xyz.normalized();
}

Intrinsics KinectCamera::colorIntrinsics() const
{
    return KinectCamera::colorIntrinsics( m_colorResolution );
}

Intrinsics KinectCamera::depthIntrinsics() const
{
    return KinectCamera::depthIntrinsics( m_depthResolution );
}

KinectCamera::Event KinectCamera::poll( NUI_SKELETON_FRAME& skeleton,
    Array2DView< uint8x4 > bgra,
    int64_t& bgraTimestamp, int& bgraFrameNumber,
    Array2DView< uint16_t > depth, Array2DView< uint16_t > playerIndex,
    int64_t& depthTimestamp, int& depthFrameNumber,
    bool& depthCapturedWithNearMode,
    int waitIntervalMilliseconds )
{
    const int NUM_EVENTS = 3;
    HANDLE eventHandles[ NUM_EVENTS ] =
    {
        m_hNextSkeletonEvent,
        m_hNextColorFrameEvent,
        m_hNextDepthFrameEvent
    };
    KinectCamera::Event eventEnums[ NUM_EVENTS ] =
    {
        KinectCamera::Event::SKELETON,
        KinectCamera::Event::COLOR,
        KinectCamera::Event::DEPTH
    };

    DWORD waitMultipleResult = WaitForMultipleObjects( NUM_EVENTS,
        eventHandles, FALSE, waitIntervalMilliseconds );
    if( waitMultipleResult == WAIT_TIMEOUT )
    {
        return KinectCamera::Event::TIMEOUT;
    }

    int eventIndex = waitMultipleResult - WAIT_OBJECT_0;
    if( eventIndex >= NUM_EVENTS )
    {
        return KinectCamera::Event::TIMEOUT;
    }
    KinectCamera::Event e = eventEnums[ eventIndex ];
    switch( e )
    {
        case KinectCamera::Event::SKELETON:
        {
            if( handleGetSkeletonFrame( skeleton ) )
            {
                return e;
            }
        }
        case KinectCamera::Event::COLOR:
        {
            if( handleGetColorFrame( bgra, bgraTimestamp, bgraFrameNumber ) )
            {
                return e;
            }
        }
        case KinectCamera::Event::DEPTH:
        {
            if( handleGetExtendedDepthFrame( depth, playerIndex,
                depthTimestamp, depthFrameNumber, depthCapturedWithNearMode ) )
            {
                return e;
            }
        }
        default:
            return KinectCamera::Event::FAILED;
    }
}

KinectCamera::Event KinectCamera::pollColorAndExtendedDepth( Array2DView< uint8x4 > bgra,
    int64_t& bgraTimestamp, int& bgraFrameNumber,
    Array2DView< uint16_t > depth, Array2DView< uint16_t > playerIndex,
    int64_t& depthTimestamp, int& depthFrameNumber,
    bool& depthCapturedWithNearMode,
    int waitIntervalMilliseconds )
{
    HANDLE eventHandles[] = { m_hNextColorFrameEvent, m_hNextDepthFrameEvent };
    DWORD waitMultipleResult = WaitForMultipleObjects( 2, eventHandles, TRUE,
        waitIntervalMilliseconds );
    if( waitMultipleResult == WAIT_TIMEOUT )
    {
        return KinectCamera::Event::TIMEOUT;
    }

    // Succeeded in waiting on all of them.
    if( waitMultipleResult >= WAIT_OBJECT_0 &&
        waitMultipleResult < WAIT_OBJECT_0 + 2 )
    {
        bool rgbSucceeded = handleGetColorFrame( bgra, bgraTimestamp,
            bgraFrameNumber );
        bool depthSucceeded = handleGetExtendedDepthFrame( depth, playerIndex,
            depthTimestamp, depthFrameNumber, depthCapturedWithNearMode );
        if( rgbSucceeded && depthSucceeded )
        {
            return KinectCamera::Event::RGBD;
        }
    }
    return KinectCamera::Event::FAILED;
}

KinectCamera::Event KinectCamera::pollColor( Array2DView< uint8x4 > bgra,
    int64_t& timestamp, int& frameNumber,
    int waitIntervalMilliseconds )
{
    DWORD waitResult = WaitForSingleObject( m_hNextColorFrameEvent,
        waitIntervalMilliseconds );

    if( waitResult == WAIT_OBJECT_0 )
    {
        bool succeeded = handleGetColorFrame( bgra, timestamp, frameNumber );
        if( succeeded )
        {
            return KinectCamera::Event::COLOR;
        }
        else
        {
            return KinectCamera::Event::FAILED;
        }
    }
    else
    {
        return KinectCamera::Event::TIMEOUT;
    }
}

KinectCamera::Event KinectCamera::pollExtendedDepth( Array2DView< uint16_t > depth,
    Array2DView< uint16_t > playerIndex,
    int64_t& timestamp, int& frameNumber, bool& capturedWithNearMode,
    int waitIntervalMilliseconds )
{
    DWORD waitResult = WaitForSingleObject( m_hNextDepthFrameEvent,
        waitIntervalMilliseconds );
    if( waitResult == WAIT_OBJECT_0 )
    {
        bool succeeded = handleGetExtendedDepthFrame( depth, playerIndex,
            timestamp, frameNumber, capturedWithNearMode );
        if( succeeded )
        {
            return KinectCamera::Event::DEPTH;
        }
        else
        {
            return KinectCamera::Event::FAILED;
        }
    }
    else
    {
        return KinectCamera::Event::TIMEOUT;
    }
}

KinectCamera::Event KinectCamera::pollPackedDepth( Array2DView< uint16_t > packedDepth,
    int waitIntervalMilliseconds )
{
    DWORD waitResult = WaitForSingleObject( m_hNextDepthFrameEvent,
        waitIntervalMilliseconds );

    if( waitResult == WAIT_OBJECT_0 )
    {
        bool succeeded = handleGetPackedDepthFrame( packedDepth );
        if( succeeded )
        {
            return KinectCamera::Event::DEPTH;
        }
        else
        {
            return KinectCamera::Event::FAILED;
        }
    }
    else
    {
        return KinectCamera::Event::TIMEOUT;
    }
}

#if KINECT1X_ENABLE_SPEECH
bool KinectCamera::pollSpeech( QString& phrase, float& confidence, int waitInterval )
{
    HRESULT hr = E_FAIL;
    SPEVENT curEvent;
    ULONG fetched;

    m_pContext->WaitForNotifyEvent( waitInterval );
    m_pContext->GetEvents( 1, &curEvent, &fetched );

    if( fetched != 0 )
    {
        switch( curEvent.eEventId )
        {
        case SPEI_RECOGNITION:
        {
            if( curEvent.elParamType == SPET_LPARAM_IS_OBJECT )
            {
                // this is an ISpRecoResult
                ISpRecoResult* result = reinterpret_cast< ISpRecoResult* >( curEvent.lParam );
                SPPHRASE* pPhrase;
                WCHAR* pwszText;

                hr = result->GetPhrase( &pPhrase );
                if( SUCCEEDED( hr ) )
                {
                    hr = result->GetText( SP_GETWHOLEPHRASE, SP_GETWHOLEPHRASE, TRUE, &pwszText, NULL );
                    if( SUCCEEDED( hr ) )
                    {
                        // confidence = pPhrase->pElements->ActualConfidence;
                        confidence = pPhrase->pElements->SREngineConfidence;
                        // apparently, the properties form a tree...
                        // but the tree is NULL
                        // confidence = pPhrase->pProperties->SREngineConfidence;
                        phrase = QString::fromUtf16( reinterpret_cast< const uint16_t* >( pwszText ) );
                        ::CoTaskMemFree( pwszText );
                    }
                    ::CoTaskMemFree( pPhrase );
                }
            }
            break;
        }
        case SPEI_SOUND_START: break;
        case SPEI_SOUND_END: break;
        case SPEI_START_INPUT_STREAM: break;
        default:
            printf("Unknown event id: %d\r\n", curEvent.eEventId);
            break;
        }
    }

    return SUCCEEDED( hr );
}
#endif

bool KinectCamera::isUsingColor() const
{
    return m_usingColor;
}

bool KinectCamera::isUsingDepth() const
{
    return m_usingDepth;
}

bool KinectCamera::isUsingPlayerIndex() const
{
    return m_usingPlayerIndex;
}

bool KinectCamera::isUsingSkeleton() const
{
    return m_usingSkeleton;
}

KinectCamera::KinectCamera( int deviceIndex ) :
    m_deviceIndex( deviceIndex )
{

}

HRESULT KinectCamera::initialize( DWORD nuiFlags,
    NUI_IMAGE_RESOLUTION colorResolution,
    NUI_IMAGE_RESOLUTION depthResolution )
{
    // create an instance
    INuiSensor* pSensor;
    HRESULT hr = NuiCreateSensorByIndex( m_deviceIndex, &pSensor );
    if( SUCCEEDED( hr ) )
    {
        hr = pSensor->NuiInitialize( nuiFlags );
        if( SUCCEEDED( hr ) )
        {
            m_pSensor = pSensor;

            m_usingColor = ( nuiFlags & NUI_INITIALIZE_FLAG_USES_COLOR ) != 0;

            m_usingDepth =
            (
                ( nuiFlags & NUI_INITIALIZE_FLAG_USES_DEPTH ) ||
                ( nuiFlags & NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX )
            ) != 0;

            m_usingSkeleton =
                ( nuiFlags & NUI_INITIALIZE_FLAG_USES_SKELETON ) != 0;
            m_usingPlayerIndex =
                ( nuiFlags & NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX )
                != 0;
            if( m_usingPlayerIndex && !m_usingSkeleton )
            {
                fprintf( stderr, "Warning: player index is requested but skeleton tracking is not enabled: enabling...\n" );
                m_usingSkeleton = true;
            }

            m_usingAudio = ( nuiFlags & NUI_INITIALIZE_FLAG_USES_AUDIO ) != 0;

            if( SUCCEEDED( hr ) && m_usingColor )
            {
                hr = initializeColorStream( colorResolution );
            }

            if( SUCCEEDED( hr ) && m_usingDepth )
            {
                hr = initializeDepthStream( depthResolution,
                    m_usingPlayerIndex );
            }

            // if depth is enabled, initialize skeleton tracking
            if( SUCCEEDED( hr ) && m_usingDepth && m_usingSkeleton )
            {
                hr = initializeSkeletonTracking();
            }

#if KINECT1X_ENABLE_SPEECH
            if( SUCCEEDED( hr ) && m_usingAudio )
            {
                hr = initializeSpeechRecognition( recognizedPhrases );
            }
#endif
        }
    }

    return hr;
}

HRESULT KinectCamera::initializeSkeletonTracking()
{
    const DWORD flags = 0; // Currently ignored by API.

    // Enable skeleton tracking.
    m_hNextSkeletonEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
    return m_pSensor->NuiSkeletonTrackingEnable( m_hNextSkeletonEvent, flags );
}

HRESULT KinectCamera::initializeColorStream( NUI_IMAGE_RESOLUTION resolution )
{
    const DWORD flags = 0; // Currently ignored by the API.
    const DWORD frameLimit = 2; // How many frames to buffer.

    m_colorResolution = resolution;
    m_colorResolutionPixels = kinectutils::toVector2i( resolution );

    // Enable color stream.
    m_hNextColorFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
    return m_pSensor->NuiImageStreamOpen
    (
        NUI_IMAGE_TYPE_COLOR,
        resolution,
        flags,
        frameLimit,
        m_hNextColorFrameEvent,
        &m_hColorStreamHandle
    );
}

HRESULT KinectCamera::initializeDepthStream( NUI_IMAGE_RESOLUTION resolution,
    bool trackPlayerIndex )
{
    NUI_IMAGE_TYPE imageType = trackPlayerIndex ?
        NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX :
        imageType = NUI_IMAGE_TYPE_DEPTH;

    m_depthResolution = resolution;
    m_depthResolutionPixels = kinectutils::toVector2i( resolution );

    const DWORD flags = 0; // Currently ignored by the API.
    const DWORD frameLimit = 2; // How many frames to buffer.

    m_hNextDepthFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
    m_hDepthStreamHandle = NULL;
    HRESULT hr = m_pSensor->NuiImageStreamOpen
    (
        imageType,
        resolution,
        flags,
        frameLimit,
        m_hNextDepthFrameEvent,
        &m_hDepthStreamHandle
    );

    if( SUCCEEDED( hr ) )
    {
        hr = NuiImageStreamGetImageFrameFlags( m_hDepthStreamHandle,
            &m_depthStreamFlags );
    }

    return hr;
}

#if KINECT1X_ENABLE_SPEECH
HRESULT KinectCamera::initializeSpeechRecognition( QVector< QString > recognizedPhrases )
{
    CoInitialize( NULL );
    HRESULT hr = initializeAudio();
    if( SUCCEEDED( hr ) )
    {
        WAVEFORMATEX wfxOut = { WAVE_FORMAT_PCM, 1, 16000, 32000, 2, 16, 0 };
        DMO_MEDIA_TYPE mt = { 0 };

        // Set DMO output format
        hr = MoInitMediaType( &mt, sizeof( WAVEFORMATEX ) );
        if( SUCCEEDED( hr ) )
        {
            mt.majortype = MEDIATYPE_Audio;
            mt.subtype = MEDIASUBTYPE_PCM;
            mt.lSampleSize = 0;
            mt.bFixedSizeSamples = TRUE;
            mt.bTemporalCompression = FALSE;
            mt.formattype = FORMAT_WaveFormatEx;
            memcpy( mt.pbFormat, &wfxOut, sizeof( WAVEFORMATEX ) );

            hr = m_pDMO->SetOutputType( 0, &mt, 0 );
            if( SUCCEEDED( hr ) )
            {
                MoFreeMediaType( &mt );

                // Allocate streaming resources. This step is optional. If it is not called here, it
                // will be called when first time ProcessInput() is called. However, if you want to
                // get the actual frame size being used, it should be called explicitly here.
                hr = m_pDMO->AllocateStreamingResources();
                if( SUCCEEDED( hr ) )
                {
                    // Get actually frame size being used in the DMO. (optional, do as you need)
                    int iFrameSize;
                    PROPVARIANT pvFrameSize;
                    PropVariantInit( &pvFrameSize );
                    hr = m_pPS->GetValue( MFPKEY_WMAAECMA_FEATR_FRAME_SIZE, &pvFrameSize );
                    if( SUCCEEDED( hr ) )
                    {
                        iFrameSize = pvFrameSize.lVal;
                        PropVariantClear( &pvFrameSize );

                        // allocate output buffer
                        m_pKS = new KinectStream( m_pDMO, wfxOut.nSamplesPerSec * wfxOut.nBlockAlign );
                        hr = m_pKS->QueryInterface( __uuidof( IStream ), reinterpret_cast< void** >( &( m_pStream ) ) );

                        // Initialize speech recognition
                        // TODO: check hr
                        // TODO: dynamically change grammar?
                        hr = CoCreateInstance( CLSID_SpInprocRecognizer, NULL, CLSCTX_INPROC_SERVER, __uuidof(ISpRecognizer), reinterpret_cast< void** >( &( m_pRecognizer ) ) );
                        hr = CoCreateInstance( CLSID_SpStream, NULL, CLSCTX_INPROC_SERVER, __uuidof(ISpStream), reinterpret_cast< void** >( &( m_pSpStream ) ) );
                        hr = m_pSpStream->SetBaseStream( m_pStream, SPDFID_WaveFormatEx, &wfxOut );
                        hr = m_pRecognizer->SetInput( m_pSpStream, FALSE );
                        hr = SpFindBestToken( SPCAT_RECOGNIZERS,L"Language=409;Kinect=True", NULL, &( m_pEngineToken ) );
                        hr = m_pRecognizer->SetRecognizer( m_pEngineToken );
                        hr = m_pRecognizer->CreateRecoContext( &( m_pContext ) );
                        hr = m_pContext->CreateGrammar(1, &( m_pGrammar ) );

                        // Populate recognition grammar
                        // See: http://msdn.microsoft.com/en-us/library/ms717885(v=vs.85).aspx
                        // Using all peers
                        hr = initializePhrases( recognizedPhrases );

                        // Start recording
                        hr = m_pKS->StartCapture();

                        // Start the recognition
                        hr = m_pRecognizer->SetRecoState( SPRST_ACTIVE_ALWAYS );
                        hr = m_pContext->SetInterest( SPFEI( SPEI_RECOGNITION ) | SPFEI( SPEI_SOUND_START ) | SPFEI( SPEI_SOUND_END ), SPFEI( SPEI_RECOGNITION ) | SPFEI( SPEI_SOUND_START ) | SPFEI( SPEI_SOUND_END ) );
                        hr = m_pContext->SetAudioOptions( SPAO_RETAIN_AUDIO, NULL, NULL );
                        hr = m_pContext->Resume( 0 );
                    }
                }
            }
        }
    }
    else
    {
        fprintf( stderr, "Unable to initialize Kinect Audio.\n" );
    }

    return hr;
}

HRESULT KinectCamera::initializePhrases( QVector< QString > recognizedPhrases )
{
    if( recognizedPhrases.count() < 1 )
    {
        return S_OK;
    }

    // find longest string
    int maxLength = recognizedPhrases[0].length();
    for( int i = 1; i < recognizedPhrases.count(); ++i )
    {
        int len = recognizedPhrases[i].length();
        if( len > maxLength )
        {
            maxLength = len;
        }
    }

    // create the rule and add
    SPSTATEHANDLE hTopLevelRule;
    HRESULT hr = m_pGrammar->GetRule( L"TopLevel", 0, SPRAF_TopLevel | SPRAF_Active, TRUE, &hTopLevelRule );

    for( int i = 0; i < recognizedPhrases.count(); ++i )
    {
        if( SUCCEEDED( hr ) )
        {
            hr = m_pGrammar->AddWordTransition( hTopLevelRule, NULL,
                reinterpret_cast< LPCWSTR >( recognizedPhrases[i].utf16() ),
                L" ", SPWT_LEXICAL, 1.0f, NULL );
        }
    }

    if( SUCCEEDED( hr ) )
    {
        hr = m_pGrammar->Commit( 0 );
        if( SUCCEEDED( hr ) )
        {
            hr = m_pGrammar->SetRuleState( NULL, NULL, SPRS_ACTIVE );
        }
    }
    return hr;
}

HRESULT KinectCamera::initializeAudio()
{
    //LPCWSTR szOutputFile = L"AECout.wav";
    //TCHAR szOutfileFullName[ MAX_PATH ];

    // DMO initialization
    INuiAudioBeam* pAudio = NULL;
    HRESULT hr = NuiGetAudioSource( &pAudio );
    if( SUCCEEDED( hr ) )
    {
        hr = pAudio->QueryInterface( IID_IMediaObject, reinterpret_cast< void** >( &m_pDMO ) );
        if( SUCCEEDED( hr ) )
        {
            hr = pAudio->QueryInterface( IID_IPropertyStore, reinterpret_cast< void** >( &m_pPS ) );
            pAudio->Release();

            PROPVARIANT pvSysMode;
            PropVariantInit( &pvSysMode );
            pvSysMode.vt = VT_I4;
            //   SINGLE_CHANNEL_AEC = 0
            //   OPTIBEAM_ARRAY_ONLY = 2
            //   OPTIBEAM_ARRAY_AND_AEC = 4
            //   SINGLE_CHANNEL_NSAGC = 5
            pvSysMode.lVal = 4;
            hr = m_pPS->SetValue( MFPKEY_WMAAECMA_SYSTEM_MODE, pvSysMode );
            if( SUCCEEDED( hr ) )
            {
                PropVariantClear( &pvSysMode );

#if 0
                wprintf( L"Play a song, e.g. in Windows Media Player, to perform echo cancellation using\nsound from speakers.\n" );
                DWORD dwRet = GetFullPathName( szOutputFile, (DWORD)ARRAYSIZE(szOutfileFullName), szOutfileFullName,NULL);
                if( dwRet == 0 )
                {
                    wprintf( L"Sound output could not be written.\n" );
                }

                wprintf( L"Sound output will be written to file: \n%s\n", szOutfileFullName );

                // NOTE: Need to wait 4 seconds for device to be ready right after initialization
                DWORD dwWait = 4;
                while( dwWait > 0 )
                {
                    wprintf( L"Device will be ready for listening in %d second(s).\r", dwWait );
                    --dwWait;
                    ::Sleep(1000);
                }
                wprintf( L"\n" );
#endif
            }
        }
    }

    return hr;
}
#endif

bool KinectCamera::handleGetSkeletonFrame( NUI_SKELETON_FRAME& skeleton )
{
    bool foundSkeleton = false;

    HRESULT hr = m_pSensor->NuiSkeletonGetNextFrame( 0, &skeleton );
    if( SUCCEEDED( hr ) )
    {
        for( int i = 0; i < NUI_SKELETON_COUNT; i++ )
        {
            if( skeleton.SkeletonData[i].eTrackingState == NUI_SKELETON_TRACKED )
            {
                foundSkeleton = true;
            }
        }
    }

    if( foundSkeleton )
    {
        // smooth data
// TODO(jiawen): allow this
#if 0
        NUI_TRANSFORM_SMOOTH_PARAMETERS smooth;
        smooth.fSmoothing = 1.0f; // max smoothing
        smooth.fCorrection = 1.0f; // max correction
        smooth.fPrediction = 5.0f; // # Predicted frames
        smooth.fJitterRadius = 0.1f; // 10 cm instead of 5
        smooth.fMaxDeviationRadius = 0.1f; // 10 cm instead of 4

        m_pSensor->NuiTransformSmooth( &skeletonFrame, &smooth );
#else
        m_pSensor->NuiTransformSmooth( &skeleton, NULL );
#endif
    }

    return foundSkeleton;
}

bool KinectCamera::handleGetColorFrame( Array2DView< uint8x4 > bgra,
    int64_t& timestamp, int& frameNumber )
{
    if( bgra.isNull() )
    {
        return false;
    }
    if( bgra.size() != m_colorResolutionPixels )
    {
        return false;
    }

    NUI_IMAGE_FRAME imageFrame;
    HRESULT hr = m_pSensor->NuiImageStreamGetNextFrame
    (
        m_hColorStreamHandle,
        0,
        &imageFrame
    );
    if( FAILED( hr ) )
    {
        return false;
    }

    NUI_LOCKED_RECT lockedRect;
    hr = imageFrame.pFrameTexture->LockRect( 0, &lockedRect, NULL, 0 );
    bool valid = SUCCEEDED( hr ) && ( lockedRect.Pitch != 0 );
    if( valid )
    {
        // Input is BGRA, with A = 0
        Array2DView< const uint8x4 > srcBGR0( lockedRect.pBits,
            m_colorResolutionPixels,
            { sizeof( uint8x4 ), lockedRect.Pitch } );
        copy( srcBGR0, bgra );

        timestamp = imageFrame.liTimeStamp.QuadPart;
        frameNumber = imageFrame.dwFrameNumber;
    }
    imageFrame.pFrameTexture->UnlockRect( 0 );
    m_pSensor->NuiImageStreamReleaseFrame( m_hColorStreamHandle, &imageFrame );

    return valid;
}

bool KinectCamera::handleGetPackedDepthFrame( Array2DView< uint16_t > packedDepth )
{
    // If depth is null, do nothing.
    if( packedDepth.isNull() )
    {
        return false;
    }
    // Check that sizes match.
    if( packedDepth.notNull() &&
        packedDepth.size() != m_depthResolutionPixels )
    {
        return false;
    }

    NUI_IMAGE_FRAME imageFrame;
    NUI_LOCKED_RECT lockedRect;
    bool valid;

    HRESULT hr = m_pSensor->NuiImageStreamGetNextFrame
    (
        m_hDepthStreamHandle,
        0,
        &imageFrame
    );
    if( FAILED( hr ) )
    {
        return false;
    }

    if( packedDepth.notNull() )
    {
        hr = imageFrame.pFrameTexture->LockRect( 0, &lockedRect, NULL, 0 );
        valid = SUCCEEDED( hr ) && ( lockedRect.Pitch != 0 );
        if( valid )
        {
            Array2DView< const uint16_t > src
            (
                lockedRect.pBits,
                m_depthResolutionPixels,
                { sizeof( uint16_t ), lockedRect.Pitch }
            );
            copy( src, packedDepth );
        }
        imageFrame.pFrameTexture->UnlockRect( 0 );
        m_pSensor->NuiImageStreamReleaseFrame( m_hDepthStreamHandle,
            &imageFrame );
    }

    return valid;
}

bool KinectCamera::handleGetExtendedDepthFrame( Array2DView< uint16_t > depth,
    Array2DView< uint16_t > playerIndex,
    int64_t& timestamp, int& frameNumber, bool& capturedWithNearMode )
{
    // If both are null, then do nothing.
    if( depth.isNull() && playerIndex.isNull() )
    {
        return false;
    }
    // If you ask for a buffer but its size does not match, return false.
    if( depth.notNull() && depth.size() != m_depthResolutionPixels )
    {
        return false;
    }
    if( playerIndex.notNull() &&
        playerIndex.size() != m_depthResolutionPixels )
    {
        return false;
    }

    NUI_IMAGE_FRAME imageFrame;
    HRESULT hr = m_pSensor->NuiImageStreamGetNextFrame
    (
        m_hDepthStreamHandle,
        0,
        &imageFrame
    );
    if( FAILED( hr ) )
    {
        return false;
    }

    INuiFrameTexture* pTexture;
    BOOL nearMode;
    hr = m_pSensor->NuiImageFrameGetDepthImagePixelFrameTexture
    (
        m_hDepthStreamHandle,
        &imageFrame,
        &nearMode,
        &pTexture
    );

    if( FAILED( hr ) )
    {
        m_pSensor->NuiImageStreamReleaseFrame( m_hDepthStreamHandle,
            &imageFrame );
        return false;
    }

    NUI_LOCKED_RECT lockedRect;
    hr = pTexture->LockRect( 0, &lockedRect, NULL, 0 );
    bool valid = SUCCEEDED( hr ) && ( lockedRect.Pitch != 0 );
    if( valid )
    {
        Array2DView< NUI_DEPTH_IMAGE_PIXEL > srcView
        (
            lockedRect.pBits,
            m_depthResolutionPixels,
            { sizeof( NUI_DEPTH_IMAGE_PIXEL ), lockedRect.Pitch }
        );

        if( depth.notNull() )
        {
            Array2DView< const uint16_t > srcDepthView =
                componentView< const uint16_t, NUI_DEPTH_IMAGE_PIXEL >
                ( srcView, offsetof( NUI_DEPTH_IMAGE_PIXEL, depth ) );
            copy( srcDepthView, depth );
        }

        if( isUsingPlayerIndex() && playerIndex.notNull() )
        {
            Array2DView< const uint16_t > srcPlayerIndexView =
                componentView< const uint16_t, NUI_DEPTH_IMAGE_PIXEL >
                ( srcView, offsetof( NUI_DEPTH_IMAGE_PIXEL, playerIndex ) );
            copy( srcPlayerIndexView, playerIndex );
        }

        timestamp = imageFrame.liTimeStamp.QuadPart;
        frameNumber = imageFrame.dwFrameNumber;
        capturedWithNearMode = ( nearMode != 0 );
    }
    pTexture->UnlockRect( 0 );
    m_pSensor->NuiImageStreamReleaseFrame( m_hDepthStreamHandle, &imageFrame );

    return valid;
}

} } // kinect1x, libcgt
