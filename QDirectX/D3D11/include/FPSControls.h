#pragma once

#include <common/BasicTypes.h>
#include <math/Arithmetic.h>
#include <math/MathUtils.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Matrix3f.h>

#include "XboxController.h"

class PerspectiveCamera;
class QMouseEvent;

struct FPSMouseParameters
{
	bool invertX;
	bool invertY;

	// this should be positive
	Vector3f translationPerPixel;
	
	// these should be positive
	// if flip is desired, change invertX and invertY
	float yawRadiansPerPixel;
	float pitchRadiansPerPixel;
	
	float fovRadiansPerMouseWheelDelta;

	FPSMouseParameters( bool invertX = false, bool invertY = true,
		const Vector3f& translationPerPixel = Vector3f( 0.05f, 0.05f, 0.05f ),
		float yawRadiansPerPixel = MathUtils::degreesToRadians( 0.25f ),
		float pitchRadiansPerPixel = MathUtils::degreesToRadians( 0.25f ),
		float fovRadiansPerMouseWheelDelta = MathUtils::degreesToRadians( 0.01f ) );
};

struct FPSKeyboardParameters
{
	float translationPerKeyPress;

	FPSKeyboardParameters( float translationPerKeyPress = 0.25f );
};

struct FPSXboxGamepadParameters
{
	bool invertX;
	bool invertY;

	float translationPerTick;

	float yawRadiansPerTick;
	float pitchRadiansPerTick;

	float fovRadiansPerTick;

	FPSXboxGamepadParameters( bool invertX = false, bool invertY = true,
		float translationPerTick = 1e-6f,
		float yawRadiansPerTick = MathUtils::degreesToRadians( -1.0f / 32000.0f ),
		float pitchRadiansPerTick = MathUtils::degreesToRadians( 1.0f / 32000.0f ),
		float fovRadiansPerTick = 0.0002f );
};

class FPSControls
{
public:

	FPSControls( PerspectiveCamera* pCamera,
		const FPSMouseParameters& mouseParameters = FPSMouseParameters(),
		const FPSKeyboardParameters& keyboardParameters = FPSKeyboardParameters(),
		const FPSXboxGamepadParameters& xboxGamepadParameters = FPSXboxGamepadParameters() );

	FPSMouseParameters& mouseParameters();
	FPSKeyboardParameters& keyboardParameters();
	FPSXboxGamepadParameters& xboxGamepadParameters();

	Vector3f upVector() const;
	void setUpVector( const Vector3f& y );
 
	void handleXboxController( XboxController* pXboxController );

	void handleMousePressEvent( QMouseEvent* event );
	void handleMouseMoveEvent( QMouseEvent* event );
	void handleMouseReleaseEvent( QMouseEvent* event );

private:

	void computeMouseRotation( Qt::MouseButtons buttons, const Vector2f& delta );
	void computeMouseTranslation( Qt::MouseButtons buttons, const Vector2f& delta );

	void computeXboxTranslation( XINPUT_GAMEPAD* pGamepad );
	void computeXboxRotation( XINPUT_GAMEPAD* pGamepad );
	// TODO: put an exponential curve on the fov, so it approaches but never gets to 0 or 180
	void computeXboxFoV( XINPUT_GAMEPAD* pGamepad );

	void applyTranslation( float dx, float dy, float dz );
	void applyRotation( float yaw, float pitch );

	PerspectiveCamera* m_pCamera;

	FPSMouseParameters m_mouseParameters;
	FPSKeyboardParameters m_keyboardParameters;
	FPSXboxGamepadParameters m_xboxGamepadParameters;

	Matrix3f m_groundPlaneToWorld; // goes from a coordinate system where the specified up vector is (0,1,0) to the world
	Matrix3f m_worldToGroundPlane;

	bool m_mouseIsDown;
	Vector2i m_previousMouseXY;
};
