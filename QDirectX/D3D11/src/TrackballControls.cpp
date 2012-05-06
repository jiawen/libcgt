#include "TrackballControls.h"

#include <QMouseEvent>
#include <geometry/GeometryUtils.h>

TrackballControls::TrackballControls( const Vector3f& sceneCenter ) :

	m_sceneCenter( sceneCenter )

{

}

Vector3f TrackballControls::sceneCenter() const
{
	return m_sceneCenter;
}

void TrackballControls::setSceneCenter()
{
	m_sceneCenter = sceneCenter();
}

void TrackballControls::handleMousePressEvent( QMouseEvent* event, const Vector2i& screenSize,
	PerspectiveCamera& camera )
{
	Vector2f xy;
	xy.x = event->x() + 0.5f;
	xy.y = ( screenSize.y - event->y() - 1 ) + 0.5f;

	Vector3f eye = camera.eye();
	Vector3f dir = camera.pixelToDirection( xy, screenSize );

	float t;
	bool intersected = GeometryUtils::raySphereIntersection( eye, dir, sceneCenter(), sphereRadius(), &t );
	if( intersected )
	{
		m_mouseIsDown = true;
		m_mouseMovePoint = eye + t * dir;
	}
}

void TrackballControls::handleMouseMoveEvent( QMouseEvent* event, const Vector2i& screenSize,
	PerspectiveCamera& camera )
{
	Vector2f xy;
	xy.x = event->x() + 0.5f;
	xy.y = ( screenSize.y - event->y() - 1 ) + 0.5f;

	Vector3f eye = m_mouseDownCamera.eye();
	Vector3f dir = m_mouseDownCamera.pixelToDirection( xy, screenSize );

	float t;
	bool intersected = GeometryUtils::raySphereIntersection( eye, dir, sceneCenter(), sphereRadius(), &t );
	if( intersected )
	{
		m_mouseDownPoint = eye + t * dir;
		applyRotation( camera );
	}
}

void TrackballControls::handleMouseReleaseEvent( QMouseEvent* event )
{

}

float TrackballControls::sphereRadius()
{
	float d = ( sceneCenter() - m_mouseDownCamera.eye() ).norm();
	return d * tan( m_mouseDownCamera.halfFovYRadians() );	
}

void TrackballControls::applyRotation( PerspectiveCamera& camera )
{
	// compute rotation matrix
	Vector3f v0 = ( m_mouseDownPoint - sceneCenter() ).normalized();
	Vector3f v1 = ( m_mouseMovePoint - sceneCenter() ).normalized();

	Vector3f axis = Vector3f::cross( v0, v1 );
	float sinTheta = axis.norm();
	float cosTheta = Vector3f::dot( v0, v1 );
	float theta = atan2f( sinTheta, cosTheta );

	Matrix3f rot = Matrix3f::rotation( axis, theta );

	Vector3f eye = sceneCenter() + rot * ( m_mouseDownCamera.eye() - sceneCenter() );
	Vector3f up = rot * m_mouseDownCamera.up();
	camera.setLookAt( eye, sceneCenter(), up );
}