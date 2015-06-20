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
    const GLFrustum& frustum )
{
    setFrustum( frustum );
    setLookAt( eye, center, up );
}

// virtual
Matrix4f OrthographicCamera::projectionMatrix() const
{
    return Matrix4f::orthographicProjection
    (
        m_frustum.left, m_frustum.right,
        m_frustum.bottom, m_frustum.top,
        m_frustum.zNear, m_frustum.zFar,
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
    outputTextStream << "left " << m_frustum.left << "\n";
    outputTextStream << "right " << m_frustum.right << "\n";
    outputTextStream << "bottom " << m_frustum.bottom << "\n";
    outputTextStream << "top " << m_frustum.top << "\n";
    outputTextStream << "zNear " << m_frustum.zNear << "\n";
    outputTextStream << "zFar " << m_frustum.zFar << "\n";

    outputFile.close();
    return true;
}