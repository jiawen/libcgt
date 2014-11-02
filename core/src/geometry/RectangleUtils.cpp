#include "geometry/RectangleUtils.h"

// static
void RectangleUtils::writeScreenAlignedTriangleStrip(
		Array1DView< Vector4f > positions,
		Array1DView< Vector2f > textureCoordinates,
		const Rect2f& positionRectangle,
		float z, float w,
		const Rect2f& textureRectangle )
{
	writeScreenAlignedTriangleStripPositions(
		positions, positionRectangle, z, w );
	writeScreenAlignedTriangleStripTextureCoordinates(
		textureCoordinates, textureRectangle );
}

// static
void RectangleUtils::writeScreenAlignedTriangleStripPositions(
		Array1DView< Vector4f > positions,
		const Rect2f& rect,
		float z, float w )
{
	positions[ 0 ] = Vector4f( rect.left(), rect.bottom(), z, w );
	positions[ 1 ] = Vector4f( rect.right(), rect.bottom(), z, w );
	positions[ 2 ] = Vector4f( rect.left(), rect.top(), z, w );
	positions[ 3 ] = Vector4f( rect.right(), rect.top(), z, w );
}

// 	static
void RectangleUtils::writeScreenAlignedTriangleStripTextureCoordinates(
		Array1DView< Vector2f > textureCoordinates,
		const Rect2f& rect )
{
	textureCoordinates[ 0 ] = Vector2f( rect.left(), rect.bottom() );
	textureCoordinates[ 1 ] = Vector2f( rect.right(), rect.bottom() );
	textureCoordinates[ 2 ] = Vector2f( rect.left(), rect.top() );
	textureCoordinates[ 3 ] = Vector2f( rect.right(), rect.top() );
}
