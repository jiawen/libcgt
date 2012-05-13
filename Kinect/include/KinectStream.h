#pragma once

#include <windows.h>

#include <ObjIdl.h>
#include <dmo.h>
#include <avrt.h>

#include <stack>
#include <queue>
#include <NuiApi.h>

#include <tchar.h>

#define SAFE_ARRAYDELETE(p) {if (p) delete[] (p); (p) = NULL;}
#define SAFE_RELEASE(p) {if (NULL != p) {(p)->Release(); (p) = NULL;}}

#define CHECK_RET(hr, message) if (FAILED(hr)) { printf("%s: %08X\n", message, hr); goto exit;}
#define CHECKHR(x) hr = x; if (FAILED(hr)) {printf("%d: %08X\n", __LINE__, hr); goto exit;}
#define CHECK_ALLOC(pb, message) if (NULL == pb) { puts(message); goto exit;}
#define CHECK_BOOL(b, message) if (!b) { hr = E_FAIL; puts(message); goto exit;}

class CStaticMediaBuffer : public IMediaBuffer {
public:
   CStaticMediaBuffer() {}
   CStaticMediaBuffer(BYTE *pData, ULONG ulSize, ULONG ulData) :
      m_pData(pData), m_ulSize(ulSize), m_ulData(ulData), m_cRef(1) {}
   STDMETHODIMP_(ULONG) AddRef() { return 2; }
   STDMETHODIMP_(ULONG) Release() { return 1; }
   STDMETHODIMP QueryInterface(REFIID riid, void **ppv) {
      if (riid == IID_IUnknown) {
         AddRef();
         *ppv = (IUnknown*)this;
         return NOERROR;
      }
      else if (riid == IID_IMediaBuffer) {
         AddRef();
         *ppv = (IMediaBuffer*)this;
         return NOERROR;
      }
      else
         return E_NOINTERFACE;
   }
   STDMETHODIMP SetLength(DWORD ulLength) {m_ulData = ulLength; return NOERROR;}
   STDMETHODIMP GetMaxLength(DWORD *pcbMaxLength) {*pcbMaxLength = m_ulSize; return NOERROR;}
   STDMETHODIMP GetBufferAndLength(BYTE **ppBuffer, DWORD *pcbLength) {
      if (ppBuffer) *ppBuffer = m_pData;
      if (pcbLength) *pcbLength = m_ulData;
      return NOERROR;
   }
   void Init(BYTE *pData, ULONG ulSize, ULONG ulData) {
        m_pData = pData;
        m_ulSize = ulSize;
        m_ulData = ulData;
    }

   HRESULT CopyFrom(CStaticMediaBuffer *other)
   {
       if (other == NULL) return E_INVALIDARG;
       if (other->m_ulData > this->m_ulSize) return E_INVALIDARG;

       memcpy(this->m_pData, other->m_pData, other->m_ulData);
       this->m_ulData = other->m_ulData;

       return S_OK;
   }
protected:
   BYTE *m_pData;
   ULONG m_ulSize;
   ULONG m_ulData;
   ULONG m_cRef;
};

HRESULT WriteToFile(HANDLE hFile, void* p, DWORD cb);
HRESULT WriteWaveHeader(HANDLE hFile, WAVEFORMATEX *pWav, DWORD *pcbWritten);
HRESULT FixUpChunkSizes(HANDLE hFile, DWORD cbHeader, DWORD cbAudioData);

class KinectStream : public IStream
{
private:
    UINT _cRef;
    IMediaObject *_pKinectDmo;
    UINT _cbOutputBufferLen;
    HANDLE _hStopEvent;
    HANDLE _hDataReady;
    HANDLE _hCaptureThread;
    static const UINT NUM_BUFFERS = 20;
    std::stack<CStaticMediaBuffer*> _writeBufferStack;
    std::queue<CStaticMediaBuffer*> _readBufferQueue;
    CStaticMediaBuffer *_curWriteBuffer;
    CStaticMediaBuffer *_curReadBuffer;
    ULONG _curReadBufferIndex;
    ULONG _bytesRead;
    CRITICAL_SECTION _csLock;

    static DWORD WINAPI     CaptureThread(LPVOID pParam);
    DWORD WINAPI            CaptureThread();
    CStaticMediaBuffer* GetWriteBuffer();
    void ReleaseBuffer(CStaticMediaBuffer* pBuffer);
    void ReleaseAllBuffers();
    HRESULT QueueCapturedData(BYTE *pData, UINT cbData);
    HRESULT QueueCapturedBuffer(CStaticMediaBuffer *pBuffer);
    HRESULT ReadOneBuffer(BYTE **ppbData, ULONG* pcbData, ULONG *pcbRead);

    BOOL IsCapturing()
    {
        return (_hStopEvent != NULL) && (WaitForSingleObject(_hStopEvent,0) != WAIT_OBJECT_0);
    }

public:
    /////////////////////////////////////////////
    // KinectStream methods
    KinectStream(IMediaObject *pKinectDmo, UINT cbOutputBufferLen) :
      _cRef(1),
      _cbOutputBufferLen(cbOutputBufferLen),
      _curWriteBuffer(NULL),
      _curReadBuffer(NULL),
      _curReadBufferIndex(0),
      _bytesRead(0),
      _hStopEvent(NULL),
      _hDataReady(NULL),
      _hCaptureThread(NULL)
    {
        pKinectDmo->AddRef();
        _pKinectDmo = pKinectDmo;
        InitializeCriticalSection(&_csLock);
    }

    ~KinectStream()
    {
        SAFE_RELEASE(_pKinectDmo);
        DeleteCriticalSection(&_csLock);
    }

    HRESULT StartCapture();
    HRESULT StopCapture();

    /////////////////////////////////////////////
    // IUnknown methods
    STDMETHODIMP_(ULONG) AddRef() { return InterlockedIncrement(&_cRef); }
    STDMETHODIMP_(ULONG) Release()
    {
        UINT ref = InterlockedDecrement(&_cRef);
        if (ref == 0)
        {
            delete this;
        }
        return ref;
    }
    STDMETHODIMP QueryInterface(REFIID riid, void **ppv)
    {
        if (riid == IID_IUnknown)
        {
            AddRef();
            *ppv = (IUnknown*)this;
            return S_OK;
        }
        else if (riid == IID_IStream)
        {
            AddRef();
            *ppv = (IStream*)this;
            return S_OK;
        }
        else
        {
            return E_NOINTERFACE;
        }
    }

    /////////////////////////////////////////////
    // IStream methods
    STDMETHODIMP Read(void *,ULONG,ULONG *);
    STDMETHODIMP Write(const void *,ULONG,ULONG *);
    STDMETHODIMP Seek(LARGE_INTEGER,DWORD,ULARGE_INTEGER *);
    STDMETHODIMP SetSize(ULARGE_INTEGER);
    STDMETHODIMP CopyTo(IStream *,ULARGE_INTEGER,ULARGE_INTEGER *,ULARGE_INTEGER *);
    STDMETHODIMP Commit(DWORD);
    STDMETHODIMP Revert();
    STDMETHODIMP LockRegion(ULARGE_INTEGER,ULARGE_INTEGER,DWORD);
    STDMETHODIMP UnlockRegion(ULARGE_INTEGER,ULARGE_INTEGER,DWORD);
    STDMETHODIMP Stat(STATSTG *,DWORD);
    STDMETHODIMP Clone(IStream **);
};