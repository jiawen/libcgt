#include "KinectCameraImpl.h"

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

namespace libcgt { namespace camera_wrappers { namespace kinect1x {

// Convert a NUI_IMAGE_RESOLUTION to a numerical resolution in pixels.
// Returns (0, 0) on NUI_IMAGE_RESOLUTION_INVALID.
Vector2i toVector2i( NUI_IMAGE_RESOLUTION resolution )
{
    switch( resolution )
    {
    case NUI_IMAGE_RESOLUTION_80x60:
        return{ 80, 60 };
        break;
    case NUI_IMAGE_RESOLUTION_320x240:
        return{ 320, 240 };
        break;
    case NUI_IMAGE_RESOLUTION_640x480:
        return{ 640, 480 };
        break;
    case NUI_IMAGE_RESOLUTION_1280x960:
        return{ 1280, 960 };
        break;
    default:
        // Invalid resolution.
        return{ 0, 0 };
    }
}

// static
std::vector< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX > > KinectCameraImpl::s_jointIndicesForBones;
// static
std::map< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX >, int > KinectCameraImpl::s_boneIndicesForJoints;

// static
const std::vector< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX > >& KinectCameraImpl::jointIndicesForBones()
{
    if( KinectCameraImpl::s_jointIndicesForBones.size() == 0 )
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
const std::map< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX >, int >& KinectCameraImpl::boneIndicesForJoints()
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
KinectCameraImpl::~KinectCameraImpl()
{
    close();
}

bool KinectCameraImpl::isValid() const
{
    return m_deviceIndex != -1;
}

int KinectCameraImpl::elevationAngle() const
{
    LONG degrees;
    m_pSensor->NuiCameraElevationGetAngle( &degrees );
    return static_cast< int >( degrees );
}

bool KinectCameraImpl::setElevationAngle( int degrees )
{
    HRESULT hr = m_pSensor->NuiCameraElevationSetAngle( degrees );
    return SUCCEEDED( hr );
}

bool KinectCameraImpl::isNearModeEnabled() const
{
    return( ( m_depthStreamFlags & NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE ) != 0 );
}

bool KinectCameraImpl::setNearModeEnabled( bool b )
{
    // Only do something if the new state is different.
    if( b ^ isNearModeEnabled() )
    {
        // NuiImageStreamSetImageFrameFlags will fail with E_INVALID_ARG if
        // depth was not enabled.
        DWORD flags;
        if( b )
        {
            flags = m_depthStreamFlags |
                NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE;
        }
        else
        {
            flags = m_depthStreamFlags &
                ~NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE;
        }

        HRESULT hr = NuiImageStreamSetImageFrameFlags(
            m_hDepthStreamHandle, flags );
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

bool KinectCameraImpl::setInfraredEmitterEnabled( bool b )
{
    HRESULT hr = m_pSensor->NuiSetForceInfraredEmitterOff( !b );
    return SUCCEEDED( hr );
}

bool KinectCameraImpl::rawAccelerometerReading( Vector4f& reading ) const
{
    Vector4* pReading = reinterpret_cast< Vector4*> ( &reading );
    HRESULT hr = m_pSensor->NuiAccelerometerGetCurrentReading( pReading );
    return SUCCEEDED( hr );
}

Vector3f KinectCameraImpl::upVectorFromAccelerometer() const
{
    Vector4f reading;
    Vector4* pReading = reinterpret_cast< Vector4* >( &reading );
    HRESULT hr = m_pSensor->NuiAccelerometerGetCurrentReading( pReading );
    return -reading.xyz.normalized();
}

Vector2i KinectCameraImpl::colorResolution() const
{
    return m_colorResolutionPixels;
}

Intrinsics KinectCameraImpl::colorIntrinsics() const
{
    return KinectCamera::colorIntrinsics( m_colorResolutionPixels );
}

Vector2i KinectCameraImpl::depthResolution() const
{
    return m_depthResolutionPixels;
}

Intrinsics KinectCameraImpl::depthIntrinsics() const
{
    return KinectCamera::depthIntrinsics( m_depthResolutionPixels );
}

// TODO(jiawen): restore event waiting abilities
#if 0
KinectCamera::Event KinectCameraImpl::pollOne(
    KinectCamera::Frame& frame,
    bool useExtendedDepth, int waitIntervalMilliseconds )
{
    frame.colorUpdated = false;
    frame.depthUpdated = false;
    frame.skeletonUpdated = false;

    DWORD nHandles = static_cast< DWORD >( m_events.size() );
    DWORD waitMultipleResult = WaitForMultipleObjects(
        nHandles, m_events.data(), FALSE,
        waitIntervalMilliseconds );
    if( waitMultipleResult == WAIT_TIMEOUT )
    {
        return KinectCamera::Event::TIMEOUT;
    }

    int eventIndex = waitMultipleResult - WAIT_OBJECT_0;
    if( eventIndex >= static_cast< int >( nHandles ) )
    {
        return KinectCamera::Event::TIMEOUT;
    }

    if( handleEvent( frame, useExtendedDepth, eventIndex ) )
    {
        return KinectCamera::Event::OK;
    }
    else
    {
        return KinectCamera::Event::FAILED;
    }
}
#endif

bool KinectCameraImpl::pollOne( KinectCamera::Frame& frame,
    bool useExtendedDepth, int waitIntervalMilliseconds )
{
    frame.colorUpdated = false;
    frame.depthUpdated = false;
    frame.infraredUpdated = false;
    frame.skeletonUpdated = false;

    if( colorFormat() == NUI_IMAGE_TYPE_COLOR )
    {
        frame.colorUpdated = handleGetColorFrame( waitIntervalMilliseconds,
            frame.color,
            frame.colorTimestamp, frame.colorFrameNumber );
        return frame.colorUpdated;
    }
    else if( isUsingDepth() )
    {
        if( useExtendedDepth )
        {
            frame.depthUpdated = handleGetExtendedDepthFrame(
                waitIntervalMilliseconds,
                frame.extendedDepth, frame.playerIndex,
                frame.depthTimestamp, frame.depthFrameNumber,
                frame.depthCapturedWithNearMode );
        }
        else
        {
            frame.depthUpdated = handleGetPackedDepthFrame(
                waitIntervalMilliseconds,
                frame.packedDepth,
                frame.depthTimestamp, frame.depthFrameNumber );
        }
        return frame.depthUpdated;
    }
    else if( colorFormat() == NUI_IMAGE_TYPE_COLOR_INFRARED )
    {
        frame.infraredUpdated = handleGetInfraredFrame(
            waitIntervalMilliseconds,
            frame.infrared,
            frame.infraredTimestamp, frame.infraredFrameNumber );
        return frame.infraredUpdated;
    }
	return false;
}

bool KinectCameraImpl::pollAll( KinectCamera::Frame& frame,
    bool useExtendedDepth, int waitIntervalMilliseconds )
{
    frame.colorUpdated = false;
    frame.depthUpdated = false;
    frame.infraredUpdated = false;
    frame.skeletonUpdated = false;
    bool allSucceeded = true;

    if( isUsingColor() )
    {
        if( colorFormat() == NUI_IMAGE_TYPE_COLOR )
        {
            frame.colorUpdated = handleGetColorFrame( waitIntervalMilliseconds,
                frame.color,
                frame.colorTimestamp, frame.colorFrameNumber );
        }
        else if( colorFormat() == NUI_IMAGE_TYPE_COLOR_INFRARED )
        {
            frame.colorUpdated = handleGetInfraredFrame(
                waitIntervalMilliseconds,
                frame.infrared,
                frame.colorTimestamp, frame.colorFrameNumber );
        }
        allSucceeded &= frame.colorUpdated;
    }

    if( isUsingDepth() )
    {
        if( useExtendedDepth )
        {
            frame.depthUpdated = handleGetExtendedDepthFrame(
                waitIntervalMilliseconds,
                frame.extendedDepth, frame.playerIndex,
                frame.depthTimestamp, frame.depthFrameNumber,
                frame.depthCapturedWithNearMode );
        }
        else
        {
            frame.depthUpdated = handleGetPackedDepthFrame(
                waitIntervalMilliseconds,
                frame.packedDepth,
                frame.depthTimestamp, frame.depthFrameNumber );
        }
        allSucceeded &= frame.depthUpdated;
    }

#if 0
    // TODO(jiawen): wrap skeleton.
    if( isUsingSkeleton() )
    {
        frame.skeletonUpdated = handleGetSkeletonFrame(
            waitIntervalMilliseconds, frame.skeleton );
        allSucceeded &= frame.skeletonUpdated;
    }
#endif
    return allSucceeded;
}

bool KinectCameraImpl::handleEvent( KinectCamera::Frame& frame,
    bool useExtendedDepth,
    DWORD eventIndex )
{
    return false;
#if 0
    if( m_events[ eventIndex ] == m_hNextColorFrameEvent )
    {
        if( colorFormat() == NUI_IMAGE_TYPE_COLOR )
        {
            frame.colorUpdated = handleGetColorFrame( frame.bgra,
                frame.colorTimestamp, frame.colorFrameNumber );
        }
        else if( colorFormat() == NUI_IMAGE_TYPE_COLOR_INFRARED )
        {
            frame.colorUpdated = handleGetInfraredFrame( frame.infrared,
                frame.colorTimestamp, frame.colorFrameNumber );
        }
        return frame.colorUpdated;
    }
    if( m_events[ eventIndex ] == m_hNextDepthFrameEvent )
    {
        if( useExtendedDepth )
        {
            frame.depthUpdated = handleGetExtendedDepthFrame(
                frame.extendedDepth, frame.playerIndex,
                frame.depthTimestamp, frame.depthFrameNumber,
                frame.depthCapturedWithNearMode );
        }
        else
        {
            frame.depthUpdated = handleGetPackedDepthFrame(
                frame.packedDepth,
                frame.depthTimestamp, frame.depthFrameNumber );
        }
        return frame.depthUpdated;
    }
    if( m_events[ eventIndex ] == m_hNextSkeletonEvent )
    {
        frame.skeletonUpdated = handleGetSkeletonFrame(
            frame.skeleton );
        return frame.skeletonUpdated;
    }
    return false;
#endif
}

#if KINECT1X_ENABLE_SPEECH
bool KinectCameraImpl::pollSpeech( QString& phrase, float& confidence, int waitInterval )
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

bool KinectCameraImpl::isUsingColor() const
{
    return m_hNextColorFrameEvent != NULL;
}

NUI_IMAGE_TYPE KinectCameraImpl::colorFormat() const
{
    return m_colorFormat;
}

bool KinectCameraImpl::isUsingDepth() const
{
    return m_hNextDepthFrameEvent != NULL;
}

bool KinectCameraImpl::isUsingPlayerIndex() const
{
    return m_usingPlayerIndex;
}

bool KinectCameraImpl::isUsingSkeleton() const
{
    return m_hNextSkeletonEvent != NULL;
}

KinectCameraImpl::KinectCameraImpl( DWORD nuiFlags,
    NUI_IMAGE_TYPE colorFormat, NUI_IMAGE_RESOLUTION colorResolution,
    NUI_IMAGE_TYPE depthFormat, NUI_IMAGE_RESOLUTION depthResolution,
    int deviceIndex ) :
    m_deviceIndex( deviceIndex )
{
    assert( deviceIndex != -1 );
    assert( deviceIndex < KinectCamera::numDevices() );

    HRESULT hr = NuiCreateSensorByIndex( m_deviceIndex, &m_pSensor );
    if( SUCCEEDED( hr ) )
    {
        hr = m_pSensor->NuiInitialize( nuiFlags );
        if( SUCCEEDED( hr ) )
        {
            bool usingColor =
                ( nuiFlags & NUI_INITIALIZE_FLAG_USES_COLOR ) != 0;
            bool usingDepth =
            (
                ( nuiFlags & NUI_INITIALIZE_FLAG_USES_DEPTH ) ||
                ( nuiFlags & NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX )
            ) != 0;

            m_usingPlayerIndex =
                ( nuiFlags & NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX )
                != 0;

            bool usingSkeleton =
                ( nuiFlags & NUI_INITIALIZE_FLAG_USES_SKELETON ) != 0;

            if( m_usingPlayerIndex && !usingSkeleton )
            {
                fprintf( stderr, "Warning: player index is requested but "
                    "skeleton tracking is not enabled: enabling...\n" );
                usingSkeleton = true;
            }

#if KINECT1X_ENABLE_SPEECH
            m_usingAudio = ( nuiFlags & NUI_INITIALIZE_FLAG_USES_AUDIO ) != 0;
#endif

            if( SUCCEEDED( hr ) && usingColor )
            {
                hr = initializeColorStream( colorFormat, colorResolution );
            }

            if( SUCCEEDED( hr ) && usingDepth )
            {
                hr = initializeDepthStream( depthResolution,
                    m_usingPlayerIndex );
            }

            // If depth stream initialization succeeded, initialize skeleton
            // tracking if it's enabled.
            if( SUCCEEDED( hr ) && usingDepth && usingSkeleton )
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

    if( FAILED( hr ) )
    {
        close();
    }
}

HRESULT KinectCameraImpl::initializeColorStream( NUI_IMAGE_TYPE format,
    NUI_IMAGE_RESOLUTION resolution )
{
    const DWORD flags = 0; // Currently ignored by the API.
    const DWORD frameLimit = 2; // How many frames to buffer.

    m_colorFormat = format;
    m_colorResolution = resolution;
    m_colorResolutionPixels = toVector2i( resolution );

    // Enable color stream.
    m_hNextColorFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
    HRESULT hr = m_pSensor->NuiImageStreamOpen
    (
        format,
        resolution,
        flags,
        frameLimit,
        m_hNextColorFrameEvent,
        &m_hColorStreamHandle
    );

    if( SUCCEEDED( hr ) )
    {
        m_events.push_back( m_hNextColorFrameEvent );
    }
    else
    {
        CloseHandle( m_hNextColorFrameEvent );
        m_hNextColorFrameEvent = NULL;
    }

    return hr;
}

HRESULT KinectCameraImpl::initializeDepthStream( NUI_IMAGE_RESOLUTION resolution,
    bool trackPlayerIndex )
{
    NUI_IMAGE_TYPE imageType = trackPlayerIndex ?
        NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX :
        imageType = NUI_IMAGE_TYPE_DEPTH;

    m_depthResolution = resolution;
    m_depthResolutionPixels = toVector2i( resolution );

    const DWORD flags = 0; // Currently ignored by the API.
    const DWORD frameLimit = 2; // How many frames to buffer.

    m_hNextDepthFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
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

    if( SUCCEEDED( hr ) )
    {
        m_events.push_back( m_hNextDepthFrameEvent );
    }
    else
    {
        CloseHandle( m_hNextDepthFrameEvent );
        m_hNextDepthFrameEvent = NULL;
    }

    return hr;
}

HRESULT KinectCameraImpl::initializeSkeletonTracking()
{
    const DWORD flags = 0; // Currently ignored by API.

    // Enable skeleton tracking.
    m_hNextSkeletonEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
    HRESULT hr = m_pSensor->NuiSkeletonTrackingEnable( m_hNextSkeletonEvent,
        flags );

    if( SUCCEEDED( hr ) )
    {
        m_events.push_back( m_hNextSkeletonEvent );
    }
    else
    {
        CloseHandle( m_hNextSkeletonEvent );
        m_hNextSkeletonEvent = NULL;
    }

    return hr;
}

#if KINECT1X_ENABLE_SPEECH
HRESULT KinectCameraImpl::initializeSpeechRecognition( QVector< QString > recognizedPhrases )
{
    CoInitialize( NULL );
    HRESULT hr = initializeAudio();
    if( SUCCEEDED( hr ) )
    {
        WAVEFORMATEX wfxOut = { WAVE_FORMAT_PCM, 1, 16000, 32000, 2, 16, 0 };
        DMO_MEDIA_TYPE mt = {};

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

HRESULT KinectCameraImpl::initializePhrases( QVector< QString > recognizedPhrases )
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

HRESULT KinectCameraImpl::initializeAudio()
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

void KinectCameraImpl::close()
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

    m_events.clear();

    // No need to close the stream handles, they are released per frame.
    m_depthStreamFlags = 0;
    if( m_hNextDepthFrameEvent != NULL &&
        m_hNextDepthFrameEvent != INVALID_HANDLE_VALUE )
    {
        CloseHandle( m_hNextDepthFrameEvent );
        m_hNextDepthFrameEvent = NULL;
    }
    m_hDepthStreamHandle = NULL;

    if( m_hNextColorFrameEvent != NULL &&
        m_hNextColorFrameEvent != INVALID_HANDLE_VALUE )
    {
        CloseHandle( m_hNextColorFrameEvent );
        m_hNextColorFrameEvent = NULL;
    }
    m_hColorStreamHandle = NULL;

    if( m_hNextSkeletonEvent != NULL &&
        m_hNextSkeletonEvent != INVALID_HANDLE_VALUE )
    {
        CloseHandle( m_hNextSkeletonEvent );
        m_hNextSkeletonEvent = NULL;
    }

    m_depthResolution = NUI_IMAGE_RESOLUTION_INVALID;
    m_colorResolution = NUI_IMAGE_RESOLUTION_INVALID;
    m_usingPlayerIndex = false;

    if( m_pSensor != nullptr )
    {
        m_pSensor->NuiShutdown();
        m_pSensor = nullptr;
    }

    m_deviceIndex = -1;
}

// TODO(jiawen): rename to pollColor.
bool KinectCameraImpl::handleGetColorFrame( DWORD millisecondsToWait,
    Array2DView< uint8x4 > bgra,
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
        millisecondsToWait,
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

bool KinectCameraImpl::handleGetInfraredFrame( DWORD millisecondsToWait,
    Array2DView< uint16_t > ir,
    int64_t& timestamp, int& frameNumber )
{
    if( ir.isNull() )
    {
        return false;
    }
    if( ir.size() != m_colorResolutionPixels )
    {
        return false;
    }

    NUI_IMAGE_FRAME imageFrame;
    HRESULT hr = m_pSensor->NuiImageStreamGetNextFrame
    (
        m_hColorStreamHandle,
        millisecondsToWait,
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
        Array2DView< const uint16_t > src( lockedRect.pBits,
            m_colorResolutionPixels,
            { sizeof( uint16_t ), lockedRect.Pitch } );
        copy( src, ir );

        timestamp = imageFrame.liTimeStamp.QuadPart;
        frameNumber = imageFrame.dwFrameNumber;
    }
    imageFrame.pFrameTexture->UnlockRect( 0 );
    m_pSensor->NuiImageStreamReleaseFrame( m_hColorStreamHandle, &imageFrame );

    return valid;
}

bool KinectCameraImpl::handleGetExtendedDepthFrame( DWORD millisecondsToWait,
    Array2DView< uint16_t > depth, Array2DView< uint16_t > playerIndex,
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
        millisecondsToWait,
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
                componentView< const uint16_t >
                ( srcView, offsetof( NUI_DEPTH_IMAGE_PIXEL, depth ) );
            copy( srcDepthView, depth );
        }

        if( isUsingPlayerIndex() && playerIndex.notNull() )
        {
            Array2DView< const uint16_t > srcPlayerIndexView =
                componentView< const uint16_t >
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


bool KinectCameraImpl::handleGetPackedDepthFrame( DWORD millisecondsToWait,
    Array2DView< uint16_t > packedDepth,
    int64_t& timestamp, int& frameNumber )
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

            timestamp = imageFrame.liTimeStamp.QuadPart;
            frameNumber = imageFrame.dwFrameNumber;
        }
        imageFrame.pFrameTexture->UnlockRect( 0 );
        m_pSensor->NuiImageStreamReleaseFrame( m_hDepthStreamHandle,
            &imageFrame );
    }

    return valid;
}


bool KinectCameraImpl::handleGetSkeletonFrame( DWORD millisecondsToWait,
    NUI_SKELETON_FRAME& skeleton )
{
    bool foundSkeleton = false;

    HRESULT hr = m_pSensor->NuiSkeletonGetNextFrame( millisecondsToWait,
        &skeleton );
    if( SUCCEEDED( hr ) )
    {
        for( int i = 0; i < NUI_SKELETON_COUNT; i++ )
        {
            if( skeleton.SkeletonData[i].eTrackingState ==
                NUI_SKELETON_TRACKED )
            {
                foundSkeleton = true;
            }
        }
    }

    if( foundSkeleton )
    {
        // smooth data
// TODO(jiawen): just make this a separate function.
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

} } } // kinect1x, camera_wrappers libcgt
