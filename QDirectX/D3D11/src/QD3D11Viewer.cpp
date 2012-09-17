#include "QD3D11Viewer.h"

#include <QApplication>
#include <QMouseEvent>

#include <geometry/GeometryUtils.h>

#include <vecmath/Vector2f.h>
#include <vecmath/Vector4f.h>
#include <vecmath/Matrix3f.h>
#include <vecmath/Quat4f.h>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

QD3D11Viewer::QD3D11Viewer( QWidget* parent,
	
	bool useTrackballMode,

	const Vector3f& sceneCenter,
	const Vector3f& sceneUpVector ) :

	m_useTrackballMode( useTrackballMode ),
	m_fpsControls( sceneUpVector ),
	m_trackballControls( sceneCenter ),

	QD3D11Widget( parent )

{
    m_camera.setDirectX( true );
    // m_camera.setPerspective( MathUtils::degreesToRadians( 10.f ), 1.f, 4.7f, 5.4f);
    // m_camera.setPerspective( MathUtils::degreesToRadians( 50.f ), 1.f, 3.5f, 6.5f);	
	m_camera.setPerspective( MathUtils::degreesToRadians( 50.f ), 1.f, 0.01f, 10.0f );

	m_pXboxController0 = new XboxController( 0, this );
}

bool QD3D11Viewer::useTrackballMode() const
{
	return m_useTrackballMode;
}

void QD3D11Viewer::setUseTrackballMode( bool b )
{
	m_useTrackballMode = b;
}

PerspectiveCamera& QD3D11Viewer::camera()
{
	return m_camera;
}

void QD3D11Viewer::setCamera( const PerspectiveCamera& camera )
{
	m_camera = camera;	
}

FPSControls& QD3D11Viewer::fpsControls()
{
	return m_fpsControls;
}

TrackballControls& QD3D11Viewer::trackballControls()
{
	return m_trackballControls;
}

XboxController* QD3D11Viewer::xboxController0()
{
	return m_pXboxController0;
}

//////////////////////////////////////////////////////////////////////////
// Protected
//////////////////////////////////////////////////////////////////////////

void QD3D11Viewer::updateKeyboard()
{
	if ( !isActiveWindow() )
	{
		return;
	}

	if( useTrackballMode() )
	{
		return;
	}

	fpsControls().handleKeyboard( m_camera );

	emit viewpointChanged( camera(), width(), height() );

	update();
}

void QD3D11Viewer::updateXboxController()
{
	if( m_pXboxController0->isConnected() )
	{
		m_pXboxController0->sampleState();
	}	
	m_fpsControls.handleXboxController( m_pXboxController0, m_camera );

	emit viewpointChanged( camera(), width(), height() );

	update();
}

// virtual
void QD3D11Viewer::keyPressEvent( QKeyEvent* event )
{
	if( event->key() == Qt::Key_Escape ||
		event->key() == Qt::Key_Q )
	{
		qApp->quit();
	}

	emit keyPressed( event );

	update();
}

void QD3D11Viewer::mousePressEvent( QMouseEvent* event )
{
	if( useTrackballMode() )
	{
		trackballControls().handleMousePressEvent( event, screenSize(), m_camera );
	}
	else
	{
		fpsControls().handleMousePressEvent( event );
	}
}

void QD3D11Viewer::mouseMoveEvent( QMouseEvent* event )
{
	if( useTrackballMode() )
	{
		trackballControls().handleMouseMoveEvent( event, screenSize(), m_camera );
	}
	else
	{
		fpsControls().handleMouseMoveEvent( event, m_camera );
	}

	emit viewpointChanged( camera(), width(), height() );

	update();
}

void QD3D11Viewer::mouseReleaseEvent( QMouseEvent* event )
{
	if( useTrackballMode() )
	{
		trackballControls().handleMouseReleaseEvent( event );
	}
	else
	{
		fpsControls().handleMouseReleaseEvent( event );
	}
}

void QD3D11Viewer::wheelEvent( QWheelEvent * event )
{
	Q_UNUSED( event );
	//float speed = 0.002f;
	//float zoom = exp(event->delta() * speed);

 //   float fovY, aspect, zNear, zFar;
 //   m_camera.getPerspective(&fovY, &aspect, &zNear, &zFar);

 //   double h = tan(fovY * 0.5f);
 //   h /= zoom;
 //   fovY = 2.f * atan(h);

 //   m_camera.setPerspective(fovY, aspect, zNear, zFar);

 //   update();
}

void QD3D11Viewer::resizeD3D( int width, int height )
{
	m_camera.setAspect( width, height );
	
	emit viewpointChanged( camera(), width, height );
}
