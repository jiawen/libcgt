#ifndef D3D11UTILS_BOX_H
#define D3D11UTILS_BOX_H

#include <geometry/BoundingBox3f.h>

#include "DynamicVertexBuffer.h"
#include "D3D11Utils.h"

class D3D11Utils_Box
{
public:

	// writes pBuffer, starting at the vertexOffset-th vertex
	// with the contents of box
	template< typename T >
	static void writeBuffer( const BoundingBox3f& box, const Vector4f& color, std::shared_ptr< DynamicVertexBuffer > pBuffer )
	{
		T* vertexArray = pBuffer->mapForWriteDiscardAs< T >();
		D3D11Utils::writeAxisAlignedBox( box.minimum(), box.range(), vertexArray );

		for( int i = 0; i < 36; ++i )
		{
			vertexArray[i].color = color;
		}

		pBuffer->unmap();
	}
};

#endif // D3D11UTILS_BOX_H
