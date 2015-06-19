#include <vecmath/Rect2f.h>
#include <vecmath/Rect2i.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector2i.h>

#include <QPoint>
#include <QPointF>
#include <QRect>
#include <QRectF>

#include "interop_vecmath.h"

QPointF libcgt::qtinterop::vecmath::toQPointF( const Vector2f& v )
{
    return QPointF( v.x, v.y );
}

QPoint libcgt::qtinterop::vecmath::toQPoint( const Vector2i& v )
{
    return QPoint( v.x, v.y );
}

Vector2f libcgt::qtinterop::vecmath::toVector2f( const QPointF& q )
{
    return Vector2f{
        static_cast< float >( q.x() ),
        static_cast< float >( q.y() ) };
}

Vector2i libcgt::qtinterop::vecmath::toVector2i( const QPoint& q )
{
    return{ q.x(), q.y() };
}

Rect2f libcgt::qtinterop::vecmath::toRect2f( const QRectF& r )
{
    return{ static_cast< float >( r.x() ), static_cast< float >( r.y() ),
        static_cast< float >( r.width() ),
        static_cast< float >( r.height() ) };
}

Rect2i libcgt::qtinterop::vecmath::toRect2i( const QRect& r )
{
    return{ r.x(), r.y(), r.width(), r.height() };
}
