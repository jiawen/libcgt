#include "QKinect.h"

// For string IO and manipulation.
#include <strsafe.h>
#include <conio.h>

#include <common/ArrayUtils.h>
#include <imageproc/Swizzle.h>

#include "KinectUtils.h"

// static
std::vector< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX > > QKinect::s_jointIndicesForBones;
// static
std::map< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX >, int > QKinect::s_boneIndicesForJoints;

// static
int QKinect::numDevices()
{
    int nDevices;
    NuiGetSensorCount( &nDevices );

    return nDevices;
}

// static
QKinect* QKinect::create
(
    int deviceIndex,
    DWORD nuiFlags,
    NUI_IMAGE_RESOLUTION colorResolution,
    NUI_IMAGE_RESOLUTION depthResolution,
    bool usingExtendedDepth,
    QVector< QString > recognizedPhrases
)
{
    QKinect* pKinect = new QKinect( deviceIndex );

    HRESULT hr = pKinect->initialize( nuiFlags, colorResolution, depthResolution, usingExtendedDepth, recognizedPhrases );
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
const std::vector< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX > >& QKinect::jointIndicesForBones()
{
    if( QKinect::s_jointIndicesForBones.size() == 0 )
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
const std::map< std::pair< NUI_SKELETON_POSITION_INDEX, NUI_SKELETON_POSITION_INDEX >, int >& QKinect::boneIndicesForJoints()
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
QKinect::~QKinect()
{
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

int QKinect::elevationAngle() const
{
    LONG degrees;
    m_pSensor->NuiCameraElevationGetAngle( &degrees );
    return static_cast< int >( degrees );
}

bool QKinect::setElevationAngle( int degrees )
{
    HRESULT hr = m_pSensor->NuiCameraElevationSetAngle( degrees );
    return SUCCEEDED( hr );
}

bool QKinect::isNearModeEnabled() const
{
    return( ( m_depthStreamFlags & NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE ) != 0 );
}

bool QKinect::setNearModeEnabled( bool b )
{
    if( b )
    {
        m_depthStreamFlags |= NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE;
    }
    else
    {
        m_depthStreamFlags &= ~NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE;
    }

    HRESULT hr = NuiImageStreamSetImageFrameFlags( m_hDepthStreamHandle, m_depthStreamFlags );
    return SUCCEEDED( hr );
}

bool QKinect::setInfraredEmitterEnabled( bool b )
{
    HRESULT hr = m_pSensor->NuiSetForceInfraredEmitterOff( b );
    return SUCCEEDED( hr );
}

bool QKinect::rawAccelerometerReading( Vector4f& reading ) const
{
    Vector4* pReading = reinterpret_cast<Vector4*>( &reading );
    HRESULT hr = m_pSensor->NuiAccelerometerGetCurrentReading( pReading );
    return SUCCEEDED( hr );
}

Vector3f QKinect::upVectorFromAccelerometer() const
{
    Vector4f reading;
    Vector4* pReading = reinterpret_cast< Vector4* >( &reading );
    HRESULT hr = m_pSensor->NuiAccelerometerGetCurrentReading( pReading );
    return -reading.xyz.normalized();
}

QKinect::Event QKinect::pollColor( Array2DView< uint8x4 > bgra, int waitIntervalMilliseconds )
{
    DWORD waitResult = WaitForSingleObject( m_hNextColorFrameEvent,
        waitIntervalMilliseconds );

    if( waitResult == WAIT_OBJECT_0 )
    {
        bool succeeded = handleGetColorFrame( bgra );
        if( succeeded )
        {
            return QKinect::Event::COLOR;
        }
        else
        {
            return QKinect::Event::FAILED;
        }
    }
    else
    {
        return QKinect::Event::TIMEOUT;
    }
}

QKinect::Event QKinect::pollDepth( Array2DView< uint16_t > depth,
    Array2DView< uint16_t > playerIndex,
    int waitIntervalMilliseconds )
{
    DWORD waitResult = WaitForSingleObject( m_hNextDepthFrameEvent,
        waitIntervalMilliseconds );

    if( waitResult == WAIT_OBJECT_0 )
    {
        bool succeeded = handleGetDepthFrame( depth, playerIndex );
        if( succeeded )
        {
            return QKinect::Event::DEPTH;
        }
        else
        {
            return QKinect::Event::FAILED;
        }
    }
    else
    {
        return QKinect::Event::TIMEOUT;
    }
}

QKinect::Event QKinect::poll( NUI_SKELETON_FRAME& skeleton,
    Array2DView< uint8x4 > bgra,
    Array2DView< uint16_t > depth,
    Array2DView< uint16_t > playerIndex,
    int waitIntervalMilliseconds )
{
    DWORD nEvents = static_cast< DWORD >( m_eventHandles.size() );
    DWORD waitMultipleResult = WaitForMultipleObjects( nEvents, m_eventHandles.data(), FALSE, waitIntervalMilliseconds );

    if( waitMultipleResult == WAIT_TIMEOUT )
    {
        return QKinect::Event::TIMEOUT;
    }

    int eventIndex = waitMultipleResult - WAIT_OBJECT_0;
    if( eventIndex >= static_cast< int >( nEvents ) )
    {
        return QKinect::Event::TIMEOUT;
    }

    bool succeeded = false;

    QKinect::Event e = m_eventEnums[ eventIndex ];
    switch( e )
    {
    case QKinect::Event::SKELETON:
        {
            succeeded = handleGetSkeletonFrame( skeleton );
            break;
        }
    case QKinect::Event::COLOR:
        {
            succeeded = handleGetColorFrame( bgra );
            break;
        }
    case QKinect::Event::DEPTH:
        {
            succeeded = handleGetDepthFrame( depth, playerIndex );
            break;
        }
    }

    if( succeeded )
    {
        return e;
    }
    else
    {
        return QKinect::Event::FAILED;
    }
}

bool QKinect::pollSpeech( QString& phrase, float& confidence, int waitInterval )
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

bool QKinect::isUsingDepth() const
{
    return m_usingDepth;
}

bool QKinect::isUsingPlayerIndex() const
{
    return m_usingPlayerIndex;
}

bool QKinect::isUsingExtendedDepth() const
{
    return m_usingExtendedDepth;
}

QKinect::QKinect( int deviceIndex ) :

    m_deviceIndex( deviceIndex ),
    m_pSensor( nullptr ),

    m_hNextSkeletonEvent( NULL ),
    m_hNextColorFrameEvent( NULL ),
    m_hNextDepthFrameEvent( NULL ),

    m_hColorStreamHandle( NULL ),
    m_hDepthStreamHandle( NULL ),

    m_pDMO( nullptr ),
    m_pPS( nullptr ),
    m_pContext( nullptr ),
    m_pKS( nullptr ),
    m_pStream( nullptr ),
    m_pRecognizer( nullptr ),
    m_pSpStream( nullptr ),
    m_pEngineToken( nullptr ),
    m_pGrammar( nullptr )

{

}

HRESULT QKinect::initialize
(
    DWORD nuiFlags,
    NUI_IMAGE_RESOLUTION colorResolution,
    NUI_IMAGE_RESOLUTION depthResolution,
    bool usingExtendedDepth,
    QVector< QString > recognizedPhrases
)
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

            m_usingColor = ( nuiFlags & NUI_INITIALIZE_FLAG_USES_COLOR );

            m_usingDepth = ( nuiFlags & NUI_INITIALIZE_FLAG_USES_DEPTH ) ||
                ( nuiFlags & NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX );

            m_usingSkeleton = nuiFlags & NUI_INITIALIZE_FLAG_USES_SKELETON;
            m_usingPlayerIndex = nuiFlags & NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX;
            if( m_usingPlayerIndex && !m_usingSkeleton )
            {
                fprintf( stderr, "Warning: player index is requested but skeleton tracking is not enabled: enabling...\n" );
                m_usingSkeleton = true;
            }

            m_usingExtendedDepth = usingExtendedDepth;

            m_usingAudio = ( nuiFlags & NUI_INITIALIZE_FLAG_USES_AUDIO );

            if( SUCCEEDED( hr ) && m_usingColor )
            {
                hr = initializeColorStream( colorResolution );
            }

            if( SUCCEEDED( hr ) && m_usingDepth )
            {
                hr = initializeDepthStream( m_usingPlayerIndex, depthResolution, usingExtendedDepth );
            }

            // if depth is enabled, initialize skeleton tracking
            if( SUCCEEDED( hr ) && m_usingDepth && m_usingSkeleton )
            {
                hr = initializeSkeletonTracking();
            }

            if( SUCCEEDED( hr ) && m_usingAudio )
            {
                hr = initializeSpeechRecognition( recognizedPhrases );
            }
        }
    }

    return hr;
}

HRESULT QKinect::initializeSkeletonTracking()
{
    // Enable skeleton tracking
    m_hNextSkeletonEvent = CreateEvent( NULL, TRUE, FALSE, NULL );

    DWORD flags; // image frame flags

    flags = 0; // currently ignored by API
    HRESULT hr = m_pSensor->NuiSkeletonTrackingEnable( m_hNextSkeletonEvent, flags );
    if( SUCCEEDED( hr ) )
    {
        m_eventHandles.push_back( m_hNextSkeletonEvent );
        m_eventEnums.push_back( QKinect::Event::SKELETON );
    }

    return hr;
}

HRESULT QKinect::initializeColorStream( NUI_IMAGE_RESOLUTION resolution )
{
    const DWORD flags = 0; // currently ignored by the API
    const DWORD frameLimit = 2; // how many frames to buffer

    m_colorResolution = resolution;
    m_colorResolutionPixels = KinectUtils::toVector2i( resolution );

    m_hNextColorFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
    m_hColorStreamHandle = NULL;

    // Enable color stream
    HRESULT hr = m_pSensor->NuiImageStreamOpen
    (
        NUI_IMAGE_TYPE_COLOR,
        resolution,
        flags,
        frameLimit,
        m_hNextColorFrameEvent,
        &m_hColorStreamHandle
    );

    if( SUCCEEDED( hr ) )
    {
        m_eventHandles.push_back( m_hNextColorFrameEvent );
        m_eventEnums.push_back( QKinect::Event::COLOR );
    }

    return hr;
}

HRESULT QKinect::initializeDepthStream( bool trackPlayerIndex, NUI_IMAGE_RESOLUTION resolution, bool usingExtendedDepth )
{
    NUI_IMAGE_TYPE imageType;

    if( trackPlayerIndex )
    {
        imageType = NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX;
    }
    else
    {
        imageType = NUI_IMAGE_TYPE_DEPTH;
    }

    m_depthResolution = resolution;
    m_depthResolutionPixels = KinectUtils::toVector2i( resolution );

    const DWORD flags = 0; // currently ignored by the API
    const DWORD frameLimit = 2; // how many frames to buffer

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
        hr = NuiImageStreamGetImageFrameFlags( m_hDepthStreamHandle, &m_depthStreamFlags );
        if( SUCCEEDED( hr ) )
        {
            m_eventHandles.push_back( m_hNextDepthFrameEvent );
            m_eventEnums.push_back( QKinect::Event::DEPTH );
        }
    }

    return hr;
}

HRESULT QKinect::initializeSpeechRecognition( QVector< QString > recognizedPhrases )
{
    CoInitialize( NULL );
    HRESULT hr = initializeAudio();
    if( SUCCEEDED( hr ) )
    {
        WAVEFORMATEX wfxOut = { WAVE_FORMAT_PCM, 1, 16000, 32000, 2, 16, 0 };
        DMO_MEDIA_TYPE mt = { 0 };

        // Set DMO output format
        hr = MoInitMediaType(&mt, sizeof( WAVEFORMATEX ) );
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

HRESULT QKinect::initializePhrases( QVector< QString > recognizedPhrases )
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

HRESULT QKinect::initializeAudio()
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

bool QKinect::handleGetSkeletonFrame( NUI_SKELETON_FRAME& skeleton )
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

bool QKinect::handleGetColorFrame( Array2DView< uint8x4 > bgra )
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

    INuiFrameTexture* pTexture = imageFrame.pFrameTexture;
    NUI_LOCKED_RECT lockedRect;
    pTexture->LockRect( 0, &lockedRect, NULL, 0 );

    bool valid = ( lockedRect.Pitch != 0 );
    if( valid )
    {
        // Input is BGRA, with A = 0
        Array2DView< const uint8x4 > srcBGR0( lockedRect.pBits,
            m_colorResolutionPixels,
            { sizeof( uint8x4 ), lockedRect.Pitch } );

        ArrayUtils::copy( srcBGR0, bgra );
    }
    pTexture->UnlockRect( 0 );
    m_pSensor->NuiImageStreamReleaseFrame( m_hColorStreamHandle, &imageFrame );

    return valid;
}

bool QKinect::handleGetDepthFrame( Array2DView< uint16_t > depth, Array2DView< uint16_t > playerIndex )
{
    // If both are null, then do nothing.
    if( depth.isNull() && playerIndex.isNull() )
    {
        return false;
    }
    // Check that sizes match.
    if( depth.notNull() && depth.size() != m_depthResolutionPixels )
    {
        return false;
    }
    if( playerIndex.notNull() && playerIndex.size() != m_depthResolutionPixels )
    {
        return false;
    }

    NUI_IMAGE_FRAME imageFrame;
    INuiFrameTexture* pTexture;
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

    if( isUsingExtendedDepth() )
    {
        BOOL capturedUsingNearMode;
        hr = m_pSensor->NuiImageFrameGetDepthImagePixelFrameTexture
        (
            m_hDepthStreamHandle,
            &imageFrame,
            &capturedUsingNearMode,
            &pTexture
        );

        if( FAILED( hr ) )
        {
            m_pSensor->NuiImageStreamReleaseFrame( m_hDepthStreamHandle, &imageFrame );
            return false;
        }

        pTexture->LockRect( 0, &lockedRect, NULL, 0 );
        valid = ( lockedRect.Pitch != 0 );
        if( valid )
        {
            Array2DView< NUI_DEPTH_IMAGE_PIXEL > srcView(
                lockedRect.pBits,
                m_depthResolutionPixels,
                { sizeof( NUI_DEPTH_IMAGE_PIXEL ), lockedRect.Pitch } );

            if( depth.notNull() )
            {
                Array2DView< const uint16_t > srcDepthView =
                    ArrayUtils::componentView< const uint16_t, NUI_DEPTH_IMAGE_PIXEL >
                       ( srcView, offsetof( NUI_DEPTH_IMAGE_PIXEL, depth ) );
                ArrayUtils::copy( srcDepthView, depth );
            }

            if( isUsingPlayerIndex() && playerIndex.notNull() )
            {
                Array2DView< const uint16_t > srcPlayerIndexView =
                    ArrayUtils::componentView< const uint16_t, NUI_DEPTH_IMAGE_PIXEL >
                    ( srcView, offsetof( NUI_DEPTH_IMAGE_PIXEL, playerIndex ) );
                ArrayUtils::copy( srcPlayerIndexView, playerIndex );
            }
        }
        pTexture->UnlockRect( 0 );
        m_pSensor->NuiImageStreamReleaseFrame( m_hDepthStreamHandle, &imageFrame );
    }
    else
    {
        if( depth.notNull() )
        {
            pTexture = imageFrame.pFrameTexture;
            pTexture->LockRect( 0, &lockedRect, NULL, 0 );
            valid = ( lockedRect.Pitch != 0 );
            if( valid )
            {
                Array2DView< const uint16_t > srcDepthView(
                    lockedRect.pBits,
                    m_depthResolutionPixels,
                    { 2, lockedRect.Pitch } );
                ArrayUtils::copy( srcDepthView, depth );
            }
            pTexture->UnlockRect( 0 );
            m_pSensor->NuiImageStreamReleaseFrame( m_hDepthStreamHandle, &imageFrame );
        }
    }

    return valid;
}
