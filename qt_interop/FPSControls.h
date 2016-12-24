#pragma once

#include <QMouseEvent>

#include "libcgt/core/common/BasicTypes.h"
#include "libcgt/core/math/Arithmetic.h"
#include "libcgt/core/math/MathUtils.h"
#include "libcgt/core/vecmath/Matrix3f.h"
#include "libcgt/core/vecmath/Vector2i.h"
#include "libcgt/core/vecmath/Vector3f.h"

using libcgt::core::math::degreesToRadians;

#ifdef XBOX_CONTROLLER_SUPPORT
#include "XboxController.h"
#endif

class PerspectiveCamera;

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
        float yawRadiansPerPixel = degreesToRadians( 0.25f ),
        float pitchRadiansPerPixel = degreesToRadians( 0.25f ),
        float fovRadiansPerMouseWheelDelta = degreesToRadians( 0.01f ) );
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
        float yawRadiansPerTick = degreesToRadians( -1.0f / 32000.0f ),
        float pitchRadiansPerTick = degreesToRadians( 1.0f / 32000.0f ),
        float fovRadiansPerTick = 0.0002f );
};

class FPSControls
{
public:

    FPSControls( const Vector3f& upVector = Vector3f( 0, 1, 0 ),
        const FPSMouseParameters& mouseParameters = FPSMouseParameters(),
        const FPSKeyboardParameters& keyboardParameters = FPSKeyboardParameters(),
        const FPSXboxGamepadParameters& xboxGamepadParameters = FPSXboxGamepadParameters() );

    FPSMouseParameters& mouseParameters();
    FPSKeyboardParameters& keyboardParameters();
    FPSXboxGamepadParameters& xboxGamepadParameters();

    Vector3f upVector() const;
    void setUpVector( const Vector3f& y );

    void handleKeyboard( PerspectiveCamera& camera );
#ifdef XBOX_CONTROLLER_SUPPORT
    void handleXboxController( XboxController* pXboxController, PerspectiveCamera& camera );
#endif

    void handleMousePressEvent( QMouseEvent* event );
    void handleMouseMoveEvent( QMouseEvent* event, PerspectiveCamera& camera );
    void handleMouseReleaseEvent( QMouseEvent* event );

private:

    void computeMouseRotation( Qt::MouseButtons buttons, const Vector2f& delta, PerspectiveCamera& camera );
    void computeMouseTranslation( Qt::MouseButtons buttons, const Vector2f& delta, PerspectiveCamera& camera );

#ifdef XBOX_CONTROLLER_SUPPORT
    void computeXboxTranslation( XINPUT_GAMEPAD* pGamepad, PerspectiveCamera& camera );
    void computeXboxRotation( XINPUT_GAMEPAD* pGamepad, PerspectiveCamera& camera );
    // TODO: put an exponential curve on the fov, so it approaches but never gets to 0 or 180
    void computeXboxFoV( XINPUT_GAMEPAD* pGamepad, PerspectiveCamera& camera );
#endif

    void applyTranslation( float dx, float dy, float dz, PerspectiveCamera& camera );
    void applyRotation( float yaw, float pitch, PerspectiveCamera& camera );

    FPSMouseParameters m_mouseParameters;
    FPSKeyboardParameters m_keyboardParameters;
    FPSXboxGamepadParameters m_xboxGamepadParameters;

    Matrix3f m_groundPlaneToWorld; // goes from a coordinate system where the specified up vector is (0,1,0) to the world
    Matrix3f m_worldToGroundPlane;

    bool m_mouseIsDown;
    Vector2i m_previousMouseXY;
};
