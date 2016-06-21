#include "FPSControls.h"

#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cameras/PerspectiveCamera.h>
#include <geometry/GeometryUtils.h>

#ifdef WIN32
#include <Windows.h>
#endif

FPSMouseParameters::FPSMouseParameters( bool invertX, bool invertY,
    const Vector3f& translationPerPixel,
    float yawRadiansPerPixel,
    float pitchRadiansPerPixel,
    float fovRadiansPerMouseWheelDelta ) :

    invertX( invertX ),
    invertY( invertY ),

    translationPerPixel( translationPerPixel ),

    yawRadiansPerPixel( yawRadiansPerPixel ),
    pitchRadiansPerPixel( pitchRadiansPerPixel ),

    fovRadiansPerMouseWheelDelta( fovRadiansPerMouseWheelDelta )

{

}

FPSKeyboardParameters::FPSKeyboardParameters( float translationPerKeyPress ) :

    translationPerKeyPress( translationPerKeyPress )

{

}

FPSXboxGamepadParameters::FPSXboxGamepadParameters( bool invertX, bool invertY,
    float translationPerTick,
    float yawRadiansPerTick, float pitchRadiansPerTick,
    float fovRadiansPerTick ) :

    invertX( invertX ),
    invertY( invertY ),
    translationPerTick( translationPerTick ),
    yawRadiansPerTick( yawRadiansPerTick ),
    pitchRadiansPerTick( pitchRadiansPerTick ),
    fovRadiansPerTick( fovRadiansPerTick )

{

}

FPSControls::FPSControls( const Vector3f& upVector,
    const FPSMouseParameters& mouseParameters,
    const FPSKeyboardParameters& keyboardParameters,
    const FPSXboxGamepadParameters& xboxGamepadParameters ) :

    m_mouseParameters( mouseParameters ),
    m_keyboardParameters( keyboardParameters ),
    m_xboxGamepadParameters( xboxGamepadParameters )

{
    setUpVector( upVector );
}

Vector3f FPSControls::upVector() const
{
    return m_groundPlaneToWorld.getCol( 1 );
}

void FPSControls::setUpVector( const Vector3f& y )
{
    Matrix3f b = GeometryUtils::getRightHandedBasis( y );
    m_groundPlaneToWorld.setCol( 0, b.getCol( 1 ) );
    m_groundPlaneToWorld.setCol( 1, b.getCol( 2 ) );
    m_groundPlaneToWorld.setCol( 2, b.getCol( 0 ) );

    m_worldToGroundPlane = m_groundPlaneToWorld.inverse();

    // TODO: snap camera to face up when you change the up vector to something new
    //   rotate along current lookat direction?
    // TODO: reset camera
}

void FPSControls::handleKeyboard( PerspectiveCamera& camera )
{
#ifdef WIN32
    Vector3f delta;

    if( ( GetAsyncKeyState( 'W' ) & 0x8000 ) != 0 )
    {
        delta.z -= m_keyboardParameters.translationPerKeyPress;
    }

    if( ( GetAsyncKeyState( 'S' ) & 0x8000 ) != 0 )
    {
        delta.z += m_keyboardParameters.translationPerKeyPress;
    }

    if( ( GetAsyncKeyState( 'A' ) & 0x8000 ) != 0 )
    {
        delta.x -= m_keyboardParameters.translationPerKeyPress;
    }

    if( ( GetAsyncKeyState( 'D' ) & 0x8000 ) != 0 )
    {
        delta.x += m_keyboardParameters.translationPerKeyPress;
    }

    if( ( GetAsyncKeyState( 'R' ) & 0x8000 ) != 0 )
    {
        delta.y += m_keyboardParameters.translationPerKeyPress;
    }

    if( ( GetAsyncKeyState( 'F' ) & 0x8000 ) != 0 )
    {
        delta.y -= m_keyboardParameters.translationPerKeyPress;
    }

    applyTranslation( delta.x, delta.y, delta.z, camera );
#endif
}

FPSMouseParameters& FPSControls::mouseParameters()
{
    return m_mouseParameters;
}

FPSKeyboardParameters& FPSControls::keyboardParameters()
{
    return m_keyboardParameters;
}

FPSXboxGamepadParameters& FPSControls::xboxGamepadParameters()
{
    return m_xboxGamepadParameters;
}

#ifdef XBOX_CONTROLLER_SUPPORT
void FPSControls::handleXboxController( XboxController* pXboxController, PerspectiveCamera& camera )
{
    if( pXboxController->isConnected() )
    {
        XINPUT_STATE state = pXboxController->getState();
        computeXboxTranslation( &state.Gamepad, camera );
        computeXboxRotation( &state.Gamepad, camera );
        computeXboxFoV( &state.Gamepad, camera );
    }
}
#endif

void FPSControls::handleMousePressEvent( QMouseEvent* event )
{
    m_previousMouseXY.x = event->x();
    m_previousMouseXY.y = event->y();
    m_mouseIsDown = true;
}

void FPSControls::handleMouseMoveEvent( QMouseEvent* event, PerspectiveCamera& camera )
{
    Vector2i currentMouseXY( { event->x(), event->y() } );
    Vector2f delta = currentMouseXY - m_previousMouseXY;

    computeMouseRotation( event->buttons(), delta, camera );
    computeMouseTranslation( event->buttons(), delta, camera );

    m_previousMouseXY = currentMouseXY;
}

void FPSControls::handleMouseReleaseEvent( QMouseEvent* event )
{
    Q_UNUSED( event );
    m_mouseIsDown = false;
}

void FPSControls::computeMouseRotation( Qt::MouseButtons buttons, const Vector2f& delta, PerspectiveCamera& camera )
{
    if( buttons == Qt::LeftButton )
    {
        float yawSpeed = m_mouseParameters.invertX ? m_mouseParameters.yawRadiansPerPixel : -m_mouseParameters.yawRadiansPerPixel;
        float pitchSpeed = m_mouseParameters.invertY ? m_mouseParameters.pitchRadiansPerPixel : -m_mouseParameters.pitchRadiansPerPixel;

        applyRotation( yawSpeed * delta.x, pitchSpeed * delta.y, camera );
    }
}

void FPSControls::computeMouseTranslation( Qt::MouseButtons buttons, const Vector2f& delta, PerspectiveCamera& camera )
{
    if( buttons == Qt::RightButton )
    {
        float x = m_mouseParameters.translationPerPixel.x * delta.x;
        float z = m_mouseParameters.translationPerPixel.y * delta.y;
        applyTranslation( x, 0, z, camera );
    }
    else if( buttons & Qt::LeftButton && buttons & Qt::RightButton )
    {
        float y = -m_mouseParameters.translationPerPixel.y * delta.y;
        applyTranslation( 0, y, 0, camera );
    }
}

#ifdef XBOX_CONTROLLER_SUPPORT
void FPSControls::computeXboxTranslation( XINPUT_GAMEPAD* pGamepad, PerspectiveCamera& camera )
{
    int lx = pGamepad->sThumbLX;
    int ly = pGamepad->sThumbLY;

    bool move = false;
    float moveX = 0;
    float moveY = 0;
    float moveZ = 0;

    // left stick: move
    if( std::abs( lx ) > XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE  )
    {
        // stick right --> move right
        moveX = lx * m_xboxGamepadParameters.translationPerTick;
        move = true;
    }
    if( std::abs( ly ) > XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE )
    {
        // stick up --> move forward
        moveZ = -ly * m_xboxGamepadParameters.translationPerTick;
        move = true;
    }

    // dpad: up/down
    if( pGamepad->wButtons & XINPUT_GAMEPAD_DPAD_UP )
    {
        moveY = 0.01f;
        move = true;
    }
    if( pGamepad->wButtons & XINPUT_GAMEPAD_DPAD_DOWN )
    {
        moveY = -0.01f;
        move = true;
    }

    if( move )
    {
        applyTranslation( moveX, moveY, moveZ, camera );
    }
}

void FPSControls::computeXboxRotation( XINPUT_GAMEPAD* pGamepad, PerspectiveCamera& camera )
{
    bool doRotate = false;
    float yaw = 0;
    float pitch = 0;
    int rx = pGamepad->sThumbRX;
    int ry = pGamepad->sThumbRY;

    // right stick: rotate
    if( std::abs( rx ) > XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE )
    {
        yaw = rx * m_xboxGamepadParameters.yawRadiansPerTick;
        doRotate = true;
    }
    if( std::abs( ry ) > XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE )
    {
        pitch = ry * m_xboxGamepadParameters.pitchRadiansPerTick;
        doRotate = true;
    }

    if( doRotate )
    {
        if( m_xboxGamepadParameters.invertX )
        {
            yaw = -yaw;
        }
        if( m_xboxGamepadParameters.invertY )
        {
            pitch = -pitch;
        }

        applyRotation( yaw, pitch, camera );
    }
}

void FPSControls::computeXboxFoV( XINPUT_GAMEPAD* pGamepad, PerspectiveCamera& camera )
{
    ubyte lt = pGamepad->bLeftTrigger;
    ubyte rt = pGamepad->bRightTrigger;

    // do nothing if both triggers are held down
    if( lt > 0 && rt > 0 )
    {
        return;
    }

    float fov = camera.fovYRadians();

    // left trigger: zoom out
    if( lt > 0 )
    {
        fov += lt * m_xboxGamepadParameters.fovRadiansPerTick;
    }
    // right trigger: zoom in
    else
    {
        fov -= rt * m_xboxGamepadParameters.fovRadiansPerTick;
    }

    float fovMin = MathUtils::degreesToRadians( 1.0f );
    float fovMax = MathUtils::degreesToRadians( 179.0f );
    fov = MathUtils::clampToRange( fov, fovMin, fovMax );

    camera.setFovYRadians( fov );
}
#endif

void FPSControls::applyTranslation( float dx, float dy, float dz, PerspectiveCamera& camera )
{
    Vector3f eye = camera.eye();
    Vector3f x = camera.right();
    Vector3f y = camera.up();
    Vector3f z = -( camera.forward() );

    // project the y axis onto the ground plane
    //Vector3f zp = m_worldToGroundPlane * z;
    //zp[ 1 ] = 0;
    //zp = m_groundPlaneToWorld * zp;
    //zp.normalize();

    eye = eye + dx * x + dy * upVector() + dz * z;
    camera.setLookAt( eye, eye - z, y );
}

void FPSControls::applyRotation( float yaw, float pitch, PerspectiveCamera& camera )
{
    Matrix3f worldToCamera = camera.viewMatrix().getSubmatrix3x3( 0, 0 );
    Matrix3f cameraToWorld = camera.inverseViewMatrix().getSubmatrix3x3( 0, 0 );

    Vector3f eye = camera.eye();
    Vector3f y = camera.up();
    Vector3f z = -( camera.forward() );

    auto x = camera.right();

    // pitch around the local x axis
    Matrix3f pitchMatrix = Matrix3f::rotateX( pitch );

    y = cameraToWorld * pitchMatrix * worldToCamera * y;
    z = cameraToWorld * pitchMatrix * worldToCamera * z;

    // yaw around the world up vector
    Matrix3f yawMatrix = m_groundPlaneToWorld * Matrix3f::rotateY( yaw ) * m_worldToGroundPlane;
    y = yawMatrix * y;
    z = yawMatrix * z;

    camera.setLookAt( eye, eye - z, y );

    auto z2 = -( camera.forward() );
}
