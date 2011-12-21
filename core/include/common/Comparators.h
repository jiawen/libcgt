#ifndef COMPARATORS_H
#define COMPARATORS_H

#include <utility>

class Comparators
{
public:
	
	// returns true if x.second < y.second
	// Useful for sorting array indices based on distance.
	static bool indexAndDistanceLess( std::pair< int, float > x, std::pair< int, float > y );

};

#endif // COMPARATORS_H
