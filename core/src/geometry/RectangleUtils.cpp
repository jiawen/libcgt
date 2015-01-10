#include "geometry/RectangleUtils.h"

#include <algorithm>

// static
Rect2i RectangleUtils::bestFitKeepAR( const Vector2i& imageSize, const Rect2i& window )
{
    int w = std::min( window.width(), window.height() * imageSize.x / imageSize.y );
    int h = std::min( window.width() * imageSize.y / imageSize.x, window.height() );
    
    int x = window.origin().x + ( window.width() - w ) / 2;
    int y = window.origin().y + ( window.height() - h ) / 2;
    
    return{ x, y, w, h };
}
  
// static
Rect2f RectangleUtils::bestFitKeepAR( const Vector2f& imageSize, const Rect2f& window )
{
    float w = std::min( window.width(), window.height() * imageSize.x / imageSize.y );
    float h = std::min( window.width() * imageSize.y / imageSize.x, window.height() );
    
    float x = window.origin().x + 0.5f * ( window.width() - w );
    float y = window.origin().y + 0.5f * ( window.height() - h );
    
    return{ x, y, w, h };
  }

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
	positions[ 0 ] = Vector4f( rect.origin().x, rect.origin().y, z, w );
	positions[ 1 ] = Vector4f( rect.limit().x, rect.origin().y, z, w );
	positions[ 2 ] = Vector4f( rect.origin().x, rect.limit().y, z, w );
	positions[ 3 ] = Vector4f( rect.limit().x, rect.limit().y, z, w );
}

// 	static
void RectangleUtils::writeScreenAlignedTriangleStripTextureCoordinates(
		Array1DView< Vector2f > textureCoordinates,
		const Rect2f& rect )
{
	textureCoordinates[ 0 ] = Vector2f( rect.origin().x, rect.origin().y );
	textureCoordinates[ 1 ] = Vector2f( rect.limit().x, rect.origin().y );
	textureCoordinates[ 2 ] = Vector2f( rect.origin().x, rect.limit().y );
	textureCoordinates[ 3 ] = Vector2f( rect.limit().x, rect.limit().y );
}
