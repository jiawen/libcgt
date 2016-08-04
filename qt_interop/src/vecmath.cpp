#include <vecmath/Rect2f.h>
#include <vecmath/Rect2i.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector2i.h>

#include <QPoint>
#include <QPointF>
#include <QRect>
#include <QRectF>

#include "vecmath.h"

namespace libcgt { namespace qt_interop { namespace vecmath {

QPointF toQPointF( const Vector2f& v )
{
    return QPointF( v.x, v.y );
}

QPoint toQPoint( const Vector2i& v )
{
    return QPoint( v.x, v.y );
}

Vector2f toVector2f( const QPointF& q )
{
    return
    {
        static_cast< float >( q.x() ),
        static_cast< float >( q.y() )
    };
}

Vector2i toVector2i( const QPoint& q )
{
    return{ q.x(), q.y() };
}

Rect2f toRect2f( const QRectF& r )
{
    return
    {
        {
            static_cast< float >( r.x() ),
            static_cast< float >( r.y() )
        },
        {
            static_cast< float >( r.width() ),
            static_cast< float >( r.height() )
        }
    };
}

Rect2i toRect2i( const QRect& r )
{
    return
    {
        { r.x(), r.y() },
        { r.width(), r.height() }
    };
}

} } } // qimage, vecmath, libcgt
