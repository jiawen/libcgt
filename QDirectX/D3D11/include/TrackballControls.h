#pragma once

#include <cameras/PerspectiveCamera.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector3f.h>

class QMouseEvent;

class TrackballControls
{
public:

	TrackballControls( const Vector3f& sceneCenter );
	
	Vector3f sceneCenter() const;
	void setSceneCenter();

	void handleMousePressEvent( QMouseEvent* event, const Vector2i& screenSize, PerspectiveCamera& camera );
	void handleMouseMoveEvent( QMouseEvent* event, const Vector2i& screenSize, PerspectiveCamera& camera );
	void handleMouseReleaseEvent( QMouseEvent* event );

private:
	
	float sphereRadius();
	void applyRotation( PerspectiveCamera& camera );

	Vector3f m_sceneCenter;

	bool m_mouseIsDown;
	PerspectiveCamera m_mouseDownCamera;
	Vector3f m_mouseDownPoint;
	Vector3f m_mouseMovePoint;

};