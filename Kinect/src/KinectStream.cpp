#include "KinectStream.h"

#include <stdio.h>

/////////////////////////////////////////////
// KinectStream methods
HRESULT KinectStream::StartCapture()
{
    HRESULT hr = S_OK;

    _hStopEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
    _hDataReady = CreateEvent( NULL, FALSE, FALSE, NULL );
    _bytesRead = 0;

    for (UINT i = 0; i < NUM_BUFFERS; i++)
    {
        BYTE *pData = new BYTE[_cbOutputBufferLen];
        CStaticMediaBuffer *pBuf = new CStaticMediaBuffer(pData, _cbOutputBufferLen, 0);
        _writeBufferStack.push(pBuf);
    }

    _curWriteBuffer = NULL;

    _hCaptureThread = CreateThread(NULL, 0, CaptureThread, this, 0, NULL);

    return hr;
}

HRESULT KinectStream::StopCapture()
{
    HRESULT hr = S_OK;
    if ( NULL != _hStopEvent )
    {
        // Signal the thread
        SetEvent(_hStopEvent);

        // Wait for thread to stop
        if ( NULL != _hCaptureThread )
        {
            WaitForSingleObject( _hCaptureThread, INFINITE );
            CloseHandle( _hCaptureThread );
            _hCaptureThread = NULL;
        }
        CloseHandle( _hStopEvent );
        _hStopEvent = NULL;
    }

    if (NULL != _hDataReady)
    {
        CloseHandle(_hDataReady);
        _hDataReady = NULL;
    }

    return hr;
}

CStaticMediaBuffer *KinectStream::GetWriteBuffer()
{
    CStaticMediaBuffer *pBuf = NULL;

    EnterCriticalSection(&_csLock);

    //Get a free buffer if available. Otherwise, get the oldest buffer
    //from the read queue. This is a way of overwriting the oldest data
    if (_writeBufferStack.size() > 0)
    {
        pBuf = _writeBufferStack.top();
        _writeBufferStack.pop();
        pBuf->SetLength(0);
        goto exit;
    }

    if (_readBufferQueue.size() > 0)
    {
        puts("Threw away unread data\n");
        pBuf = _readBufferQueue.front();
        _readBufferQueue.pop();
        pBuf->SetLength(0);
        goto exit;
    }

exit:
    LeaveCriticalSection(&_csLock);

    return pBuf;
}


void KinectStream::ReleaseBuffer(CStaticMediaBuffer* pBuffer)
{
    if (pBuffer != NULL)
    {
        EnterCriticalSection(&_csLock);
        pBuffer->SetLength(0);
        _writeBufferStack.push(pBuffer);
        LeaveCriticalSection(&_csLock);
    }
}

void KinectStream::ReleaseAllBuffers()
{
    EnterCriticalSection(&_csLock);
    while (_readBufferQueue.size() > 0)
    {
        CStaticMediaBuffer *pBuf = _readBufferQueue.front();
        _readBufferQueue.pop();
        ReleaseBuffer(pBuf);
    }
    if (_curReadBuffer != NULL)
    {
        ReleaseBuffer(_curReadBuffer);
    }

    _curReadBufferIndex = 0;
    _curReadBuffer = NULL;
    LeaveCriticalSection(&_csLock);
}

HRESULT KinectStream::QueueCapturedData(BYTE *pData, UINT cbData)
{
    BYTE *pWriteData = NULL;
    DWORD cbWriteData = 0;
    DWORD cbMaxLength = 0;
    HRESULT hr = S_OK;

    if (cbData <= 0) goto exit;

    if (_curWriteBuffer == NULL)
    {
        _curWriteBuffer = GetWriteBuffer();
    }

    CHECK_BOOL(_curWriteBuffer != NULL, "Invalid write buffer");

    CHECKHR(_curWriteBuffer->GetBufferAndLength(&pWriteData, &cbWriteData));
    CHECKHR(_curWriteBuffer->GetMaxLength(&cbMaxLength));

    if (cbWriteData + cbData < cbMaxLength)
    {
        memcpy(pWriteData + cbWriteData, pData, cbData);
        _curWriteBuffer->SetLength(cbWriteData + cbData);
    }
    else
    {
        QueueCapturedBuffer(_curWriteBuffer);

        _curWriteBuffer = GetWriteBuffer();
        CHECKHR(_curWriteBuffer->GetBufferAndLength(&pWriteData, &cbWriteData));

        memcpy(pWriteData, pData, cbData);
        _curWriteBuffer->SetLength(cbData);
    }

exit:

    return hr;
}

HRESULT KinectStream::QueueCapturedBuffer(CStaticMediaBuffer *pBuffer)
{
    HRESULT hr = S_OK;

    EnterCriticalSection(&_csLock);
    _readBufferQueue.push(pBuffer);
    SetEvent(_hDataReady);

    LeaveCriticalSection(&_csLock);

    return hr;
}

DWORD WINAPI KinectStream::CaptureThread(LPVOID pParam)
{
    KinectStream* pthis = reinterpret_cast< KinectStream* >( pParam );
    return pthis->CaptureThread();
}

DWORD WINAPI KinectStream::CaptureThread()
{
    HANDLE mmHandle = NULL;
    DWORD mmTaskIndex = 0;
    HRESULT hr = S_OK;
    bool bContinue = true;
    BYTE *pbOutputBuffer = NULL;
    CStaticMediaBuffer outputBuffer;
    DMO_OUTPUT_DATA_BUFFER OutputBufferStruct = {0};
    OutputBufferStruct.pBuffer = &outputBuffer;
    DWORD dwStatus = 0;
    ULONG cbProduced = 0;

    // Set high priority to avoid getting preempted while capturing sound
    mmHandle = AvSetMmThreadCharacteristics(L"Audio", &mmTaskIndex);
    CHECK_BOOL(mmHandle != NULL, "failed to set thread priority\n");

    pbOutputBuffer = new BYTE[_cbOutputBufferLen];
    CHECK_ALLOC (pbOutputBuffer, "out of memory.\n");

    while( bContinue )
    {
        Sleep( 10 ); // sleep 10ms

        if( WaitForSingleObject(_hStopEvent, 0) == WAIT_OBJECT_0)
        {
            bContinue = false;
            continue;
        }

        do
        {
            outputBuffer.Init((BYTE*)pbOutputBuffer, _cbOutputBufferLen, 0);
            OutputBufferStruct.dwStatus = 0;
            hr = _pKinectDmo->ProcessOutput(0, 1, &OutputBufferStruct, &dwStatus);
            CHECK_RET (hr, "ProcessOutput failed.");

            if (hr == S_FALSE) {
                cbProduced = 0;
            } else {
                hr = outputBuffer.GetBufferAndLength(NULL, &cbProduced);
                CHECK_RET (hr, "GetBufferAndLength failed");
            }

            // Queue audio data to be read by IStream client
            QueueCapturedData(pbOutputBuffer, cbProduced);
        } while (OutputBufferStruct.dwStatus & DMO_OUTPUT_DATA_BUFFERF_INCOMPLETE);
    }

exit:
    SAFE_ARRAYDELETE(pbOutputBuffer);

    SetEvent(_hDataReady);
    AvRevertMmThreadCharacteristics(mmHandle);

    return SUCCEEDED(hr) ? 0 : 1;
}

HRESULT KinectStream::ReadOneBuffer(BYTE **ppbData, ULONG* pcbData, ULONG *pcbRead)
{
    HRESULT hr = S_OK;

    EnterCriticalSection(&_csLock);

    //Do we already have a buffer we are reading from? Otherwise grab one from the queue
    if (_curReadBuffer == NULL)
    {
        if(_readBufferQueue.size() != 0)
        {
            _curReadBuffer = _readBufferQueue.front();
            _readBufferQueue.pop();
        }
    }

    LeaveCriticalSection(&_csLock);

    if (_curReadBuffer != NULL)
    {
        //Copy as much data as we can or need
        BYTE *pData = NULL;
        DWORD dwDataLength = 0;
        hr = _curReadBuffer->GetBufferAndLength(&pData, &dwDataLength);
        CHECKHR(hr);

        ULONG cbToCopy = std::min(dwDataLength - _curReadBufferIndex, *pcbData);
        memcpy(*ppbData, pData + _curReadBufferIndex, cbToCopy);
        *ppbData = (*ppbData)+cbToCopy;
        *pcbData = (*pcbData)-cbToCopy;
        *pcbRead = cbToCopy;
        _curReadBufferIndex += cbToCopy;

        //If we are done with this buffer put it back in the queue
        if (_curReadBufferIndex >= dwDataLength)
        {
            ReleaseBuffer(_curReadBuffer);
            _curReadBuffer = NULL;
            _curReadBufferIndex = 0;

            if(_readBufferQueue.size() != 0)
            {
                _curReadBuffer = _readBufferQueue.front();
                _readBufferQueue.pop();
            }
        }
    }

exit:

    return hr;
}

/////////////////////////////////////////////
// IStream methods
STDMETHODIMP KinectStream::Read(void *pBuffer, ULONG cbBuffer, ULONG *pcbRead)
{
    HRESULT hr = S_OK;

    CHECK_BOOL(pcbRead != NULL, "NULL pcbRead pointer passed in to KinectStream::Read");

    int dataRead = 0;
    while (cbBuffer > 0 && IsCapturing())
    {
        ULONG cbRead = 0;
        hr = ReadOneBuffer((BYTE**)&pBuffer, &cbBuffer, &cbRead);
        CHECKHR(hr);

        dataRead += cbRead;

        if (_curReadBuffer == NULL) //no data, wait ...
        {
            WaitForSingleObject(_hDataReady, INFINITE);
        }
    }
    _bytesRead += dataRead;

    *pcbRead = dataRead;

exit:
    return hr;
}

STDMETHODIMP KinectStream::Write(const void *,ULONG,ULONG *)
{
    printf("Called KinectStream::Write\n");
    return E_NOTIMPL;
}

STDMETHODIMP KinectStream::Seek(LARGE_INTEGER dlibMove,DWORD dwOrigin, ULARGE_INTEGER *plibNewPosition )
{
    if (plibNewPosition != NULL)
    {
        plibNewPosition->QuadPart = _bytesRead + dlibMove.QuadPart;
    }
    return S_OK;
}

STDMETHODIMP KinectStream::SetSize(ULARGE_INTEGER)
{
    printf("Called KinectStream::SetSize\n");
    return E_NOTIMPL;
}

STDMETHODIMP KinectStream::CopyTo(IStream *,ULARGE_INTEGER,ULARGE_INTEGER *,ULARGE_INTEGER *)
{
    printf("Called KinectStream::CopyTo\n");
    return E_NOTIMPL;
}

STDMETHODIMP KinectStream::Commit(DWORD)
{
    printf("Called KinectStream::Commit\n");
    return E_NOTIMPL;
}

STDMETHODIMP KinectStream::Revert()
{
    printf("Called KinectStream::Revert\n");
    return E_NOTIMPL;
}

STDMETHODIMP KinectStream::LockRegion(ULARGE_INTEGER,ULARGE_INTEGER,DWORD)
{
    printf("Called KinectStream::LockRegion\n");
    return E_NOTIMPL;
}

STDMETHODIMP KinectStream::UnlockRegion(ULARGE_INTEGER,ULARGE_INTEGER,DWORD)
{
    printf("Called KinectStream::UnlockRegion\n");
    return E_NOTIMPL;
}

STDMETHODIMP KinectStream::Stat(STATSTG *,DWORD)
{
    printf("Called KinectStream::Stat\n");
    return E_NOTIMPL;
}

STDMETHODIMP KinectStream::Clone(IStream **)
{
    printf("Called KinectStream::Clone\n");
    return E_NOTIMPL;
}
