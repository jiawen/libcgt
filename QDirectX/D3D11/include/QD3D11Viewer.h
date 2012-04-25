#pragma once

#include <cameras/PerspectiveCamera.h>
#include <vecmath/Matrix3f.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector3f.h>

#include "QD3D11Widget.h"
#include "FPSControls.h"

class QD3D11Viewer : public QD3D11Widget
{
	Q_OBJECT

public:

	QD3D11Viewer( float keyWalkSpeed = 0.15f,
		QWidget* parent = nullptr );

	// world units per key press
	float keyWalkSpeed() const;
	void setKeyWalkSpeed( float speed );

	PerspectiveCamera& camera();
	void setCamera( const PerspectiveCamera& camera );

	XboxController* xboxController0();

	Vector3f upVector() const;
	void setUpVector( const Vector3f& y );

protected:

	// keyboard handlers
	virtual void keyPressEvent( QKeyEvent* event );

	// mouse handlers
	virtual void mousePressEvent( QMouseEvent* event );
	virtual void mouseMoveEvent( QMouseEvent* event );
	virtual void mouseReleaseEvent( QMouseEvent* event );
	virtual void wheelEvent( QWheelEvent* event );

	virtual void resizeD3D( int width, int height );

	// sample keyboard state (moves the camera)
	virtual void updateKeyboard();

	// sample xbox controller state (i.e. move the camera with thumbsticks)
	virtual void updateXboxController();

	FPSControls m_fpsControls;
	XboxController* m_pXboxController0;

private:

	void translate( float dx, float dy, float dz );

	PerspectiveCamera m_camera;

	float m_keyWalkSpeed;
};
