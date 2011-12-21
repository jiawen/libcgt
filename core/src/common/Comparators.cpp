#include "common/Comparators.h"

// static
bool Comparators::indexAndDistanceLess( std::pair< int, float > x, std::pair< int, float > y )
{
	return x.second < y.second;
}
