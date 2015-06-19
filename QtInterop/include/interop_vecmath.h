#pragma once

class Rect2f;
class Rect2i;
class Vector2f;
class Vector2i;
class QPoint;
class QPointF;
class QRect;
class QRectF;

namespace libcgt
{
namespace qtinterop
{
namespace vecmath
{
    QPointF toQPointF( const Vector2f& v );
    QPoint toQPoint( const Vector2i& v );

    // Does a static_cast double --> float.
    Vector2f toVector2f( const QPointF& q );

    Vector2i toVector2i( const QPoint& q );

    // Direct copy of x, y, width, height without any flipping up/down.
    Rect2f toRect2f( const QRectF& r );

    // Direct copy of x, y, width, height without any flipping up/down.
    Rect2i toRect2i( const QRect& r );    
}
}
}

