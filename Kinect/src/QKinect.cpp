#include "QKinect.h"

// For string input,output and manipulation
#include <strsafe.h>
#include <conio.h>

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
std::shared_ptr< QKinect > QKinect::create( int deviceIndex, QVector< QString > recognizedPhrases )
{
	std::shared_ptr< QKinect > pKinect( new QKinect );

	// create an instance
	INuiSensor* pSensor;
	HRESULT hr = NuiCreateSensorByIndex( deviceIndex, &pSensor );
	if( SUCCEEDED( hr ) )
	{
		DWORD nuiFlags = NUI_INITIALIZE_FLAG_USES_COLOR;		
		nuiFlags |= NUI_INITIALIZE_FLAG_USES_DEPTH;
		nuiFlags |= NUI_INITIALIZE_FLAG_USES_AUDIO;
		nuiFlags |= NUI_INITIALIZE_FLAG_USES_SKELETON;

		hr = pSensor->NuiInitialize( nuiFlags );
		if( SUCCEEDED( hr ) )
		{
			pKinect->m_pSensor = pSensor;
			hr = initializeSkeletonTracking( pKinect.get() );
			if( SUCCEEDED( hr ) )
			{
				hr = initializeSpeechRecognition( pKinect.get(), recognizedPhrases );
				if( SUCCEEDED( hr ) )
				{
					return pKinect;
				}
				else
				{
					fprintf( stderr, "Unable to initialize speech recognition.\n" );
				}
			}
		}

		if( pSensor != nullptr )
		{
			pSensor->NuiShutdown();
		}
	}

	return nullptr;
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

	if( m_hNextDepthFrameEvent != NULL &&
		m_hNextDepthFrameEvent != INVALID_HANDLE_VALUE )
	{
		CloseHandle( m_hNextDepthFrameEvent );
		m_hNextDepthFrameEvent = NULL;
	}

	if( m_hNextVideoFrameEvent != NULL &&
		m_hNextVideoFrameEvent != INVALID_HANDLE_VALUE )
	{
		CloseHandle( m_hNextVideoFrameEvent );
		m_hNextVideoFrameEvent = NULL;
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

int QKinect::elevationAngle()
{
	LONG degrees;
	m_pSensor->NuiCameraElevationGetAngle( &degrees );
	return static_cast< int >( degrees );
}

void QKinect::setElevationAngle( int degrees )
{
	m_pSensor->NuiCameraElevationSetAngle( degrees );	
}

int QKinect::poll( NUI_SKELETON_FRAME& skeletonFrame, Image4ub& rgba, Array2D< ushort >& depth, int waitInterval )
{
	int nEvents = 3;
	DWORD waitResult = WaitForMultipleObjects( nEvents, m_events, FALSE, waitInterval );

	// TODO: 
	// waitResult contains smallest of the indices of the events that signaled
	// can do WaitForSingleObject( others ) with timeout of 0 to see if any of the others are ready
	// and return a bitflag

	bool succeeded = ( waitResult >= WAIT_OBJECT_0 && waitResult < WAIT_OBJECT_0 + nEvents );
	if( succeeded )
	{
		int eventIndex = ( waitResult - WAIT_OBJECT_0 );
		if( eventIndex == 0 )
		{
			skeletonFrame = handleGetSkeletonFrame();
			return eventIndex;
		}
		else if( eventIndex == 1 )
		{
			succeeded = handleGetColorFrame( rgba );
			if( succeeded )
			{
				return eventIndex;
			}
		}
		else if( eventIndex == 2 )
		{
			succeeded = handleGetDepthFrame( depth );
			if( succeeded )
			{
				return eventIndex;
			}
		}
	}

	return -1;
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
						phrase = QString::fromWCharArray( pwszText );
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

QKinect::QKinect() :

	m_pSensor( nullptr ),

	m_hNextSkeletonEvent( nullptr ),
	m_hNextVideoFrameEvent( nullptr ),
	m_hVideoStreamHandle( nullptr ),

	m_pDMO( nullptr ),
	m_pPS( nullptr ),
	m_pKS( nullptr ),
	m_pStream( nullptr ),
	m_pRecognizer( nullptr ),
	m_pSpStream( nullptr ),
	m_pEngineToken( nullptr ),
	m_pGrammar( nullptr )

{
	m_events[0] = nullptr;
	m_events[1] = nullptr;
}

// static
HRESULT QKinect::initializeSkeletonTracking( QKinect* pKinect )
{
	// Enable skeleton tracking	
	HANDLE hNextSkeletonEvent = CreateEvent( NULL, TRUE, FALSE, NULL );

	DWORD flags; // image frame flags
	DWORD frameLimit; // how many frames to buffer

	flags = 0; // currently ignored by API
	HRESULT hr = pKinect->m_pSensor->NuiSkeletonTrackingEnable( hNextSkeletonEvent, flags );
	if( SUCCEEDED( hr ) )
	{
		flags = 0; // currently ignored by the API
		frameLimit = 2;

		HANDLE hNextVideoFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
		HANDLE hVideoStreamHandle = NULL;

		// Enable color stream
		hr = pKinect->m_pSensor->NuiImageStreamOpen
		(
			NUI_IMAGE_TYPE_COLOR,
			NUI_IMAGE_RESOLUTION_640x480,
			flags,
			frameLimit,
			hNextVideoFrameEvent,
			&hVideoStreamHandle
		);
				
		if( SUCCEEDED( hr ) )
		{
			flags = 0; // currently ignored by the API
			frameLimit = 2;

			HANDLE hNextDepthFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
			HANDLE hDepthStreamHandle = NULL;

			hr = pKinect->m_pSensor->NuiImageStreamOpen
			(
				NUI_IMAGE_TYPE_DEPTH,
				NUI_IMAGE_RESOLUTION_640x480,
				flags,
				frameLimit,
				hNextDepthFrameEvent,
				&hDepthStreamHandle
			);

			if( SUCCEEDED( hr ) )
			{
				pKinect->m_hNextSkeletonEvent = hNextSkeletonEvent;
				pKinect->m_events[0] = hNextSkeletonEvent;
			
				pKinect->m_hVideoStreamHandle = hVideoStreamHandle;
				pKinect->m_hNextVideoFrameEvent = hNextVideoFrameEvent;
				pKinect->m_events[1] = hNextVideoFrameEvent;

				pKinect->m_hDepthStreamHandle = hDepthStreamHandle;
				pKinect->m_hNextDepthFrameEvent = hNextDepthFrameEvent;
				pKinect->m_events[2] = hNextDepthFrameEvent;

				return hr;
			}
			// Otherwise, close the handle if it's somehow valid
			if( hNextDepthFrameEvent != NULL &&
				hNextDepthFrameEvent != INVALID_HANDLE_VALUE )
			{
				CloseHandle( hNextDepthFrameEvent );
				hNextDepthFrameEvent = NULL;
			}
		}

		// Otherwise, close the handle if it's somehow valid
		if( hNextVideoFrameEvent != NULL &&
			hNextVideoFrameEvent != INVALID_HANDLE_VALUE )
		{
			CloseHandle( hNextVideoFrameEvent );
			hNextVideoFrameEvent = NULL;
		}
	}

	if( hNextSkeletonEvent != NULL &&
		hNextSkeletonEvent != INVALID_HANDLE_VALUE )
	{
		CloseHandle( hNextSkeletonEvent );
		hNextSkeletonEvent = NULL;
	}

	return hr;
}

// static
HRESULT QKinect::initializeSpeechRecognition( QKinect* pKinect, QVector< QString > recognizedPhrases )
{
	CoInitialize( NULL );
	HRESULT hr = initializeAudio( pKinect );
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

			hr = pKinect->m_pDMO->SetOutputType( 0, &mt, 0 );
			if( SUCCEEDED( hr ) )
			{
				MoFreeMediaType( &mt );

				// Allocate streaming resources. This step is optional. If it is not called here, it
				// will be called when first time ProcessInput() is called. However, if you want to 
				// get the actual frame size being used, it should be called explicitly here.
				hr = pKinect->m_pDMO->AllocateStreamingResources();
				if( SUCCEEDED( hr ) )
				{
					// Get actually frame size being used in the DMO. (optional, do as you need)
					int iFrameSize;
					PROPVARIANT pvFrameSize;
					PropVariantInit( &pvFrameSize );
					hr = pKinect->m_pPS->GetValue( MFPKEY_WMAAECMA_FEATR_FRAME_SIZE, &pvFrameSize );
					if( SUCCEEDED( hr ) )
					{
						iFrameSize = pvFrameSize.lVal;
						PropVariantClear( &pvFrameSize );

						// allocate output buffer
						pKinect->m_pKS = new KinectStream( pKinect->m_pDMO, wfxOut.nSamplesPerSec * wfxOut.nBlockAlign );
						hr = pKinect->m_pKS->QueryInterface( __uuidof( IStream ), reinterpret_cast< void** >( &( pKinect->m_pStream ) ) );

						// Initialize speech recognition
						// TODO: check hr
						// TODO: dynamically change grammar?
						hr = CoCreateInstance( CLSID_SpInprocRecognizer, NULL, CLSCTX_INPROC_SERVER, __uuidof(ISpRecognizer), reinterpret_cast< void** >( &( pKinect->m_pRecognizer ) ) );
						hr = CoCreateInstance( CLSID_SpStream, NULL, CLSCTX_INPROC_SERVER, __uuidof(ISpStream), reinterpret_cast< void** >( &( pKinect->m_pSpStream ) ) );
						hr = pKinect->m_pSpStream->SetBaseStream( pKinect->m_pStream, SPDFID_WaveFormatEx, &wfxOut );
						hr = pKinect->m_pRecognizer->SetInput( pKinect->m_pSpStream, FALSE );
						hr = SpFindBestToken( SPCAT_RECOGNIZERS,L"Language=409;Kinect=True", NULL, &( pKinect->m_pEngineToken ) );
						hr = pKinect->m_pRecognizer->SetRecognizer( pKinect->m_pEngineToken );
						hr = pKinect->m_pRecognizer->CreateRecoContext( &( pKinect->m_pContext ) );
						hr = pKinect->m_pContext->CreateGrammar(1, &( pKinect->m_pGrammar ) );						

						// Populate recognition grammar
						// See: http://msdn.microsoft.com/en-us/library/ms717885(v=vs.85).aspx
						// Using all peers
						hr = initializePhrases( pKinect, recognizedPhrases );

						// Start recording
						hr = pKinect->m_pKS->StartCapture();

						// Start the recognition
						hr = pKinect->m_pRecognizer->SetRecoState( SPRST_ACTIVE_ALWAYS );
						hr = pKinect->m_pContext->SetInterest( SPFEI( SPEI_RECOGNITION ) | SPFEI( SPEI_SOUND_START ) | SPFEI( SPEI_SOUND_END ), SPFEI( SPEI_RECOGNITION ) | SPFEI( SPEI_SOUND_START ) | SPFEI( SPEI_SOUND_END ) );
						hr = pKinect->m_pContext->SetAudioOptions( SPAO_RETAIN_AUDIO, NULL, NULL );
						hr = pKinect->m_pContext->Resume( 0 );
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

// static
HRESULT QKinect::initializePhrases( QKinect* pKinect, QVector< QString > recognizedPhrases )
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
	
	// make a WCHAR vector of the same size
	std::vector< WCHAR > wcArray( maxLength + 2 );

	// create the rule and add
	SPSTATEHANDLE hTopLevelRule;
	HRESULT hr = pKinect->m_pGrammar->GetRule( L"TopLevel", 0, SPRAF_TopLevel | SPRAF_Active, TRUE, &hTopLevelRule );

	for( int i = 0; i < recognizedPhrases.count(); ++i )
	{
		if( SUCCEEDED( hr ) )
		{
			std::fill( wcArray.begin(), wcArray.end(), 0 );
			recognizedPhrases[i].toWCharArray( wcArray.data() );
			hr = pKinect->m_pGrammar->AddWordTransition( hTopLevelRule, NULL, wcArray.data(), L" ", SPWT_LEXICAL, 1.0f, NULL );
		}
	}

	if( SUCCEEDED( hr ) )
	{
		hr = pKinect->m_pGrammar->Commit( 0 );
		if( SUCCEEDED( hr ) )
		{
			hr = pKinect->m_pGrammar->SetRuleState( NULL, NULL, SPRS_ACTIVE );
		}
	}
	return hr;
}

// static
HRESULT QKinect::initializeAudio( QKinect* pKinect )
{
	//LPCWSTR szOutputFile = L"AECout.wav";
	//TCHAR szOutfileFullName[ MAX_PATH ];

	// DMO initialization
	INuiAudioBeam* pAudio = NULL;	
	HRESULT hr = NuiGetAudioSource( &pAudio );
	if( SUCCEEDED( hr ) )
	{
		hr = pAudio->QueryInterface( IID_IMediaObject, reinterpret_cast< void** >( &( pKinect->m_pDMO ) ) );
		if( SUCCEEDED( hr ) )
		{
			hr = pAudio->QueryInterface( IID_IPropertyStore, reinterpret_cast< void** >( &( pKinect->m_pPS ) ) );
			pAudio->Release();

			PROPVARIANT pvSysMode;
			PropVariantInit( &pvSysMode );
			pvSysMode.vt = VT_I4;
			//   SINGLE_CHANNEL_AEC = 0
			//   OPTIBEAM_ARRAY_ONLY = 2
			//   OPTIBEAM_ARRAY_AND_AEC = 4
			//   SINGLE_CHANNEL_NSAGC = 5
			pvSysMode.lVal = 4;
			hr = pKinect->m_pPS->SetValue( MFPKEY_WMAAECMA_SYSTEM_MODE, pvSysMode );
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

NUI_SKELETON_FRAME QKinect::handleGetSkeletonFrame()
{
	bool foundSkeleton = false;

	NUI_SKELETON_FRAME skeletonFrame;
	HRESULT hr = m_pSensor->NuiSkeletonGetNextFrame( 0, &skeletonFrame );
	if( SUCCEEDED( hr ) )
	{
		for( int i = 0; i < NUI_SKELETON_COUNT; i++ )
		{
			if( skeletonFrame.SkeletonData[i].eTrackingState == NUI_SKELETON_TRACKED )
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
		m_pSensor->NuiTransformSmooth( &skeletonFrame, NULL );
#endif
	}

	//emit skeletonFrameReady( skeletonFrame );
	return skeletonFrame;
}

bool QKinect::handleGetColorFrame( Image4ub& rgba )
{
	NUI_IMAGE_FRAME imageFrame;

	HRESULT hr = m_pSensor->NuiImageStreamGetNextFrame
	(
		m_hVideoStreamHandle,
		0,
		&imageFrame
	);
	if( FAILED( hr ) )
	{
		return false;
	}

	INuiFrameTexture* pTexture = imageFrame.pFrameTexture;
	NUI_SURFACE_DESC desc;
	pTexture->GetLevelDesc( 0, &desc );	

	NUI_LOCKED_RECT lockedRect;
	pTexture->LockRect( 0, &lockedRect, NULL, 0 );

	bool valid = ( lockedRect.Pitch != 0 );
	if( valid )
	{
		// TODO: rgba.resize()

		BYTE* pBuffer = ( BYTE* )( lockedRect.pBits );

		for( int y = 0; y < rgba.height(); ++y )
		{
			BYTE* pSrcRow = &( pBuffer[ y * lockedRect.Pitch ] );
			quint8* pDstRow = rgba.rowPointer( y );

			int i = 0;
			for( int x = 0; x < rgba.width(); ++x )
			{
				// input is BGRA, with A = 0
				BYTE b = pSrcRow[ i ];
				BYTE g = pSrcRow[ i + 1 ];
				BYTE r = pSrcRow[ i + 2 ];
				// BYTE a = pSrcRow[ i + 3 ];
				
				pDstRow[ i ] = r;
				pDstRow[ i + 1 ] = g;
				pDstRow[ i + 2 ] = b;
				pDstRow[ i + 3 ] = 255;

				i += 4;
			}
		}
	}
	pTexture->UnlockRect( 0 );
	m_pSensor->NuiImageStreamReleaseFrame( m_hVideoStreamHandle, &imageFrame );

	return valid;
}

bool QKinect::handleGetDepthFrame( Array2D< ushort >& depth )
{
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

	INuiFrameTexture* pTexture = imageFrame.pFrameTexture;

	NUI_LOCKED_RECT lockedRect;
	pTexture->LockRect( 0, &lockedRect, NULL, 0 );

	bool valid = ( lockedRect.Pitch != 0 );
	if( valid )
	{
		// TODO: depth.resize()

		BYTE* pBuffer = ( BYTE* )( lockedRect.pBits );		
		memcpy( depth, pBuffer, lockedRect.size );
	}
	pTexture->UnlockRect( 0 );
	m_pSensor->NuiImageStreamReleaseFrame( m_hDepthStreamHandle, &imageFrame );

	return valid;
}
