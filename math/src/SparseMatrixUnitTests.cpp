#include "CoordinateSparseMatrix.h"
#include "PARDISOSolver.h"
#include "FloatMatrix.h"

void test()
{
	// create a 5 x 3 sparse matrix in "coordinate format"
	//
	// J =
	// [  1 -1  0 ]
	// [ -1  5  0 ]
	// [  0  0  4 ]
	// [ -3  0  6 ]
	// [  0  0  4 ]
	//
	// J' =
	// [  1 -1  0 -3  0 ]
	// [ -1  5  0  0  0 ]
	// [  0  0  4  6  4 ]
	//
	// J' * J =
	// [  11   -6   -18 ]
	// [  -6   26     0 ]
	// [ -18    0    68 ]
	CoordinateSparseMatrix< float > coordJ;
	coordJ.append( 0, 0, 1 );
	coordJ.append( 1, 0, -1 );
	coordJ.append( 3, 0, -3 );
	coordJ.append( 0, 1, -1 );
	coordJ.append( 1, 1, 5 );
	coordJ.append( 2, 2, 4 );
	coordJ.append( 3, 2, 6 );
	coordJ.append( 4, 2, 4 );

	// test transposing on the coordinate format
	CoordinateSparseMatrix< float > coordJt;
	coordJ.transposed( coordJt );

	// test compress and compressTranspose:
	// convert into CSC form
	CompressedSparseMatrix< float > cscJ( GENERAL );
	CompressedSparseMatrix< float > cscJt( GENERAL );

	std::vector< int > indexMapCSCJ;
	std::vector< int > indexMapCSCJt;

	coordJ.compress( cscJ, indexMapCSCJ );
	coordJ.compressTranspose( cscJt, indexMapCSCJt );

	// test directly transposing a CSC sparse matrix
	CompressedSparseMatrix< float > cscJt2( GENERAL );
	cscJ.transposed( cscJt2 );

	FloatMatrix x( 5, 1 );
	x[0] = 1;
	x[1] = 2;
	x[2] = 3;
	x[3] = 4;
	x[4] = 5;

	// compute J' * x in five different ways

	FloatMatrix coordJ_tx;
	coordJ.multiplyTransposeVector( x, coordJ_tx );

	FloatMatrix coordJt_x;
	coordJt.multiplyVector( x, coordJt_x );

	FloatMatrix cscJ_tx;
	cscJ.multiplyTransposeVector( x, cscJ_tx );

	FloatMatrix cscJt_x;
	cscJt.multiplyVector( x, cscJt_x );

	FloatMatrix cscJt2_x;
	cscJt2.multiplyVector( x, cscJt2_x );


	CompressedSparseMatrix< float > cscJtJ( SYMMETRIC );
	CompressedSparseMatrix< float >::multiply( cscJt, cscJ, cscJtJ );

	PARDISOSolver< float, true > solver;
	solver.analyzePattern( cscJtJ );

	solver.factorize( cscJtJ );

	// b = [ 1 1 1 ]'
	// Solve J'J x = b
	// x =
	//
	//  0.30830223880597
	//	0.109608208955224
	//	0.0963152985074627

	FloatMatrix b( 3, 1, 1 );
	FloatMatrix y( 3, 1, 1 );

	solver.solve( b, y );
}
