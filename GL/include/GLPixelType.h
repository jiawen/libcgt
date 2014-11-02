#pragma once

#include <GL/glew.h>

enum class GLPixelType
{
	// Unsigned integers.
	UNSIGNED_BYTE = GL_UNSIGNED_BYTE,
	UNSIGNED_SHORT = GL_UNSIGNED_SHORT,
	UNSIGNED_INT = GL_UNSIGNED_INT,	

	// Signed integers.
	BYTE = GL_BYTE,
	SHORT = GL_SHORT,
	INT = GL_INT,

	// TODO: Packed unsigned integers.
	// GL_UNSIGNED_BYTE_

	// Float.
	HALF_FLOAT = GL_HALF_FLOAT,
	FLOAT = GL_FLOAT,

	// Depth + Stencil.
	UNSIGNED_INT_24_8 = GL_UNSIGNED_INT_24_8,

	// 64-bit packed floating point depth + stencil.
	// [ 31 30 ...         0 | 31 30 ...         8 | 7 6 ... 0 ]
	// [ float part of depth | fixed part of depth | stencil   ]
	FLOAT_32_UNSIGNED_24_8_REV = GL_FLOAT_32_UNSIGNED_INT_24_8_REV
};