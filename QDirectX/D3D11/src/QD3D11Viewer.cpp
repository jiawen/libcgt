#include "QD3D11Viewer.h"

#include <QApplication>
#include <QMouseEvent>

#include <geometry/GeometryUtils.h>

#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>
#include <vecmath/Matrix3f.h>
#include <vecmath/Quat4f.h>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

QD3D11Viewer::QD3D11Viewer( float keyWalkSpeed,
						   QWidget* parent ) :

	m_keyWalkSpeed( keyWalkSpeed ),

	m_fpsControls( &m_camera ),	

	QD3D11Widget( parent )

{
    m_camera.setDirectX( true );
    // m_camera.setPerspective( 10.f, 1.f, 4.7f, 5.4f);
    // m_camera.setPerspective( 50.f, 1.f, 3.5f, 6.5f);	
	m_camera.setPerspective( 50.f, 1.f, 0.01f, 10.0f );

	m_pXboxController0 = new XboxController( 0, this );
}

float QD3D11Viewer::keyWalkSpeed() const
{
	return m_keyWalkSpeed;
}

void QD3D11Viewer::setKeyWalkSpeed( float speed )
{
	m_keyWalkSpeed = speed;
}

PerspectiveCamera& QD3D11Viewer::camera()
{
	return m_camera;
}

void QD3D11Viewer::setCamera( const PerspectiveCamera& camera )
{
	m_camera = camera;
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
		return;

	if( ( GetAsyncKeyState( 'W' ) & 0x8000 ) != 0 )
	{
		translate( 0, 0, m_keyWalkSpeed );
	}

	if( ( GetAsyncKeyState( 'S' ) & 0x8000 ) != 0 )
	{
		translate( 0, 0, -m_keyWalkSpeed );
	}

	if( ( GetAsyncKeyState( 'A' ) & 0x8000 ) != 0 )
	{
		translate( -m_keyWalkSpeed, 0, 0 );
	}

	if( ( GetAsyncKeyState( 'D' ) & 0x8000 ) != 0 )
	{
		translate( m_keyWalkSpeed, 0, 0 );
	}

	if( ( GetAsyncKeyState( 'R' ) & 0x8000 ) != 0 )
	{
		translate( 0, m_keyWalkSpeed, 0 );
	}

	if( ( GetAsyncKeyState( 'F' ) & 0x8000 ) != 0 )
	{
		translate( 0, -m_keyWalkSpeed, 0 );
	}
}

void QD3D11Viewer::updateXboxController()
{
	m_pXboxController0->sampleState();
	m_fpsControls.handleXboxController( m_pXboxController0 );
}

// virtual
void QD3D11Viewer::keyPressEvent( QKeyEvent* event )
{
	if( event->key() == Qt::Key_Escape ||
		event->key() == Qt::Key_Q )
	{
		qApp->quit();
	}

	update();
}

void QD3D11Viewer::mousePressEvent( QMouseEvent* event )
{
	m_fpsControls.handleMousePressEvent( event );
}

void QD3D11Viewer::mouseMoveEvent( QMouseEvent* event )
{
#if 1

	m_fpsControls.handleMouseMoveEvent( event );

#else
	// this is a trackball: refactor into its own controls class

	if(event->buttons() & Qt::RightButton) //rotate
	{
		float rotSpeed = 0.005f; //radians per pixel
		Quat4f rotation;
		rotation.setAxisAngle(rotSpeed * delta.abs(), Vector3f(-delta[1], -delta[0], 0));
		Matrix3f rotMatrix = Matrix3f::rotation(rotation);
		Matrix3f viewMatrix = m_camera.getViewMatrix().getSubmatrix3x3(0, 0);
		rotMatrix = viewMatrix.transposed() * rotMatrix * viewMatrix;

		Vector3f eye, center, up;
		m_camera.getLookAt(&eye, &center, &up);
		m_camera.setLookAt(center + rotMatrix * (eye - center), center, rotMatrix * up);
	}
	else if(event->buttons() & Qt::LeftButton) //translate
	{
		float speed = 10.f;
		Vector3f screenDelta(delta[0], delta[1], 0);
		screenDelta[0] /= -double(width());
		screenDelta[1] /= double(height());
		Matrix4f iViewProjMatrix = m_camera.getInverseViewProjectionMatrix();
		Vector3f worldDelta = iViewProjMatrix.getSubmatrix3x3(0, 0) * (speed * screenDelta);

		Vector3f eye, center, up;
		m_camera.getLookAt(&eye, &center, &up);
		m_camera.setLookAt(eye + worldDelta, center + worldDelta, up);
	}
#endif

	update();
}

void QD3D11Viewer::mouseReleaseEvent( QMouseEvent* event )
{
	m_fpsControls.handleMouseReleaseEvent( event );
}

void QD3D11Viewer::wheelEvent( QWheelEvent * event )
{
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
}

//////////////////////////////////////////////////////////////////////////
// Private
//////////////////////////////////////////////////////////////////////////

void QD3D11Viewer::translate( float dx, float dy, float dz )
{
	Vector3f eye = m_camera.eye();
	Vector3f x = m_camera.right();
	Vector3f y = m_camera.up();
	Vector3f z = -( m_camera.forward() );

	// project the y axis onto the ground plane
	//Vector3f zp = m_worldToGroundPlane * z;
	//zp[ 1 ] = 0;
	//zp = m_groundPlaneToWorld * zp;
	//zp.normalize();

	// TODO: switch Camera over to have just a forward vector?
	// center is kinda stupid
	eye = eye + dx * x + dy * m_fpsControls.upVector() + dz * z;
	m_camera.setLookAt( eye, eye - z, y );
}
