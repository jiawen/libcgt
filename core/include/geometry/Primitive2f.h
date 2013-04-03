#pragma once

#include "vecmath/Rect2f.h"

// interface
class Primitive2f
{
public:

	virtual Rect2f boundingBox() = 0;

};
