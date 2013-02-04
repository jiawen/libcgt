#include <windows.h>
#include <Vfw.h>

#include <QString>

#include <vecmath/Vector2i.h>

#include "AVIInputVideoStream.h"

class AVIInputFile
{
public:

	static AVIInputFile* open( QString filename );
	virtual ~AVIInputFile();

	int numStreams() const;

	int width() const;
	int height() const;
	Vector2i size() const;
	int numFrames() const;

	// can be different for each stream
	float framesPerSecond() const;
	void framesPerSecondRational( int& numerator, int& denominator ) const;

	bool allKeyFrames() const;
	bool uncompressed() const;

	AVIInputVideoStream* openVideoStream( int streamIndex = 0 );

private:

	AVIInputFile();

	PAVIFILE m_pAVIFile;
	AVIFILEINFO m_fileInfo;
};
