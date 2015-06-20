#include "SingularValueDecomposition.h"

#include <algorithm>
#include <mkl.h>

// static
bool SingularValueDecomposition::SVD( const FloatMatrix& a, FloatMatrix& u, FloatMatrix& s, FloatMatrix& vt )
{
    int m = a.numRows();
    int n = a.numCols();

    u.resize( m, m );
    s.resize( std::min( m, n ), 1 );
    vt.resize( n, n );

    // make a copy of A
    FloatMatrix b( a );

    char jobu = 'A';
    char jobvt = 'A';

    int lda = m;
    int ldu = m;
    int ldvt = n;

    float workQuery;
    int lWork = -1;
    int info;

    // do workspace query
    sgesvd( &jobu, &jobvt, &m, &n, b.data(), &lda, s.data(), u.data(), &ldu, vt.data(), &ldvt, &workQuery, &lWork, &info );

    lWork = static_cast< int >( workQuery );
    std::vector< float > work( lWork );

    sgesvd( &jobu, &jobvt, &m, &n, b.data(), &lda, s.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lWork, &info );
    if( info < 0 )
    {
        fprintf( stderr, "SVD: Illegal parameter value: %d.\n", -info );
    }
    else if( info > 0 )
    {
        fprintf( stderr, "SVD: did not converge.\n" );
    }

    bool succeeded = ( info == 0 );
    return succeeded;
}

SingularValueDecomposition::SingularValueDecomposition( const FloatMatrix& a ) :

    m_u( a.numRows(), a.numRows() ),
    m_s( std::min( a.numRows(), a.numCols() ), 1 ),
    m_vt( a.numCols(), a.numCols() )

{
    m_valid = SingularValueDecomposition::SVD( a, m_u, m_s, m_vt );
}

bool SingularValueDecomposition::isValid() const
{
    return m_valid;
}

const FloatMatrix& SingularValueDecomposition::u() const
{
    return m_u;
}

const FloatMatrix& SingularValueDecomposition::s() const
{
    return m_s;
}

const FloatMatrix& SingularValueDecomposition::vt() const
{
    return m_vt;
}
