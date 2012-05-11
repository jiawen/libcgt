#pragma once

#include <cameras/PerspectiveCamera.h>
#include <vecmath/Matrix3f.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector3f.h>

#include "QD3D11Widget.h"
#include "FPSControls.h"
#include "TrackballControls.h"

// TODO: mouse wheel for zoom
// TODO: trackball:
//    move in/out with right mouse button
//    change up vector with middle mouse button
//    zoom in/out with wheel
//    deal with switching modes: when the camera isn't nicely facing the scene center
// 
//    keep moving when the mouse move point leaves the sphere
//      - keep previous rotation around
//      - switch to accumulating rotations?
//
//    spin around view direction when initial point is outside sphere
// TODO: add a camera path
// TODO: add effect manager built in (maybe to the widget itself)
// TODO: fitCameraToScene

class QD3D11Viewer : public QD3D11Widget
{
	Q_OBJECT

public:

	QD3D11Viewer( QWidget* parent = nullptr,
		bool useTrackballMode = true,
		const Vector3f& sceneCenter = Vector3f( 0, 0, 0 ),
		const Vector3f& sceneUpVector = Vector3f( 0, 1, 0 ) );

	bool useTrackballMode() const;
	void setUseTrackballMode( bool b );

	PerspectiveCamera& camera();
	void setCamera( const PerspectiveCamera& camera );

	FPSControls& fpsControls();
	TrackballControls& trackballControls();

	// TODO: make fpscontrols and trackballcontrols not hold a pointer to camera
	// but pass it in whenever there's a handle*?
	
	// TODO: xboxController and upVector goes into FPSControls
	XboxController* xboxController0();

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
	
	XboxController* m_pXboxController0;

private:

	PerspectiveCamera m_camera;

	bool m_useTrackballMode;

	FPSControls m_fpsControls;
	TrackballControls m_trackballControls;
};
