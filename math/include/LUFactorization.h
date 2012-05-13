#pragma once

#include <memory>
#include "FloatMatrix.h"

class LUFactorization
{
public:
				
	static std::shared_ptr< LUFactorization > LU( const FloatMatrix& a );

	// output <- inverse of A
	// returns whether inverse succeeded
	bool inverse( FloatMatrix& output );

	//property FloatMatrix^ L { FloatMatrix^ get(); }
	//property FloatMatrix^ U { FloatMatrix^ get(); }	

private:

	LUFactorization( const FloatMatrix& a );

	int m_nRowsA;
	int m_nColsA;

	FloatMatrix m_y;
	std::vector< int > m_ipiv;

	FloatMatrix m_l;
	FloatMatrix m_u;
};
