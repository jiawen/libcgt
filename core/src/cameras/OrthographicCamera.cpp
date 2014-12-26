#include "cameras/OrthographicCamera.h"

#include <QFile>
#include <QTextStream>
#include <QString>

#include <math/MathUtils.h>
#include <vecmath/Quat4f.h>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

OrthographicCamera::OrthographicCamera( const Vector3f& eye,
	const Vector3f& center,
	const Vector3f& up,
	float left, float right,
	float bottom, float top,
	float zNear, float zFar )
{
	setOrtho( left, right, bottom, top, zNear, zFar );
	setLookAt( eye, center, up );
}

void OrthographicCamera::getOrtho( float* pLeft, float* pRight, float* pBottom, float* pTop, float* pZNear, float* pZFar ) const
{
	*pLeft = m_left;
	*pRight = m_right;
	*pBottom = m_bottom;
	*pTop = m_top;
	*pZNear = m_zNear;
	*pZFar = m_zFar;
}

void OrthographicCamera::setOrtho( float left, float right, float bottom, float top, float zNear, float zFar )
{
	setFrustum( left, right, bottom, top, zNear, zFar );
}

// virtual
Matrix4f OrthographicCamera::projectionMatrix() const
{
	return Matrix4f::orthographicProjection
	(
		m_left, m_right,
		m_bottom, m_top,
		m_zNear, m_zFar,
		m_directX
	);
}

bool OrthographicCamera::saveTXT( QString filename )
{
	QFile outputFile( filename );

	// try to open the file in write only mode
	if( !( outputFile.open( QIODevice::WriteOnly ) ) )
	{
		return false;
	}

	QTextStream outputTextStream( &outputFile );
	outputTextStream.setCodec( "UTF-8" );

	outputTextStream << "eye " << m_eye[ 0 ] << " " << m_eye[ 1 ] << " " << m_eye[ 2 ] << "\n";
	outputTextStream << "center " << m_center[ 0 ] << " " << m_center[ 1 ] << " " << m_center[ 2 ] << "\n";
	outputTextStream << "up " << m_up[ 0 ] << " " << m_up[ 1 ] << " " << m_up[ 2 ] << "\n";
	outputTextStream << "left " << m_left << "\n";
	outputTextStream << "right " << m_right << "\n";
	outputTextStream << "bottom " << m_bottom << "\n";
	outputTextStream << "top " << m_top << "\n";
	outputTextStream << "zNear " << m_zNear << "\n";
	outputTextStream << "zFar " << m_zFar << "\n";	

	outputFile.close();
	return true;
}