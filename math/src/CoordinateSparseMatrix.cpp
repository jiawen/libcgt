#include "CoordinateSparseMatrix.h"

#include <cassert>
#include <algorithm>

#include <mkl_spblas.h>

#include "FloatMatrix.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

template< typename T >
CoordinateSparseMatrix< T >::CoordinateSparseMatrix() :

    m_nRows( 0 ),
    m_nCols( 0 )

{

}

template< typename T >
CoordinateSparseMatrix< T >::CoordinateSparseMatrix( int initialCapacity ) :

    m_nRows( 0 ),
    m_nCols( 0 )

{
    reserve( initialCapacity );
}

template< typename T >
CoordinateSparseMatrix< T >::CoordinateSparseMatrix( const CoordinateSparseMatrix< T >& copy ) :

    m_nRows( copy.m_nRows ),
    m_nCols( copy.m_nCols ),

    m_rowIndices( copy.m_rowIndices ),
    m_colIndices( copy.m_colIndices ),
    m_values( copy.m_values )

{

}


template< typename T >
CoordinateSparseMatrix< T >& CoordinateSparseMatrix< T >::operator = ( const CoordinateSparseMatrix< T >& copy )
{
    if( this != &copy )
    {
        m_nRows = copy.m_nRows;
        m_nCols = copy.m_nCols;

        m_rowIndices = copy.m_rowIndices;
        m_colIndices = copy.m_colIndices;
        m_values = copy.m_values;
    }

    return( *this );
}

template< typename T >
int CoordinateSparseMatrix< T >::numNonZeroes() const
{
    return static_cast< int >( m_values.size() );
}

template< typename T >
int CoordinateSparseMatrix< T >::numRows() const
{
    return m_nRows;
}

template< typename T >
int CoordinateSparseMatrix< T >::numCols() const
{
    return m_nCols;
}

template< typename T >
void CoordinateSparseMatrix< T >::append( int i, int j, const T& value )
{
    m_rowIndices.push_back( i );
    m_colIndices.push_back( j );
    m_values.push_back( value );

    if( i >= m_nRows )
    {
        m_nRows = i + 1;
    }
    if( j >= m_nCols )
    {
        m_nCols = j + 1;
    }
}

template< typename T >
void CoordinateSparseMatrix< T >::clear()
{
    m_nRows = 0;
    m_nCols = 0;
    m_rowIndices.clear();
    m_colIndices.clear();
    m_values.clear();
}

template< typename T >
SparseMatrixTriplet< T > CoordinateSparseMatrix< T >::get( int k ) const
{
    assert( k < m_values.size() );

    SparseMatrixTriplet< T > t;

    t.i = m_rowIndices[ k ];
    t.j = m_colIndices[ k ];
    t.value = m_values[ k ];

    return t;
}

template< typename T >
void CoordinateSparseMatrix< T >::reserve( int nnz )
{
    m_rowIndices.reserve( nnz );
    m_colIndices.reserve( nnz );
    m_values.reserve( nnz );
}

template< typename T >
void CoordinateSparseMatrix< T >::transpose()
{
    int nnz = numNonZeroes();

    for( int k = 0; k < nnz; ++k )
    {
        std::swap( m_rowIndices[ k ], m_colIndices[ k ] );
    }

    std::swap( m_nRows, m_nCols );
}

template< typename T >
void CoordinateSparseMatrix< T >::transposed( CoordinateSparseMatrix< T >& f ) const
{
    f = ( *this );
    f.transpose();
}

template< typename T >
void CoordinateSparseMatrix< T >::compress( CompressedSparseMatrix< T >& output ) const
{
    int m = numRows();
    int n = numCols();
    int nnz = numNonZeroes();
    output.reset( m, n, nnz );

    // copy the SparseMatrixTriplet< T > vector
    std::vector< SparseMatrixTriplet< T > > ijv( nnz );

    for( int k = 0; k < nnz; ++k )
    {
        SparseMatrixTriplet< T >& t = ijv[ k ];
        t.i = m_rowIndices[ k ];
        t.j = m_colIndices[ k ];
        t.value = m_values[ k ];
    }

    std::sort( ijv.begin(), ijv.end(), colMajorLess );

    compressCore( ijv, output );

    // populate structure map
    auto& structureMap = output.structureMap();
    for( int k = 0; k < nnz; ++k )
    {
        const SparseMatrixTriplet< T >& t = ijv[k];
        structureMap[ std::make_pair( t.i, t.j ) ] = k;
    }
}

template< typename T >
void CoordinateSparseMatrix< T >::compress( CompressedSparseMatrix< T >& output, std::vector< int >& indexMap ) const
{
    compress( output );

    const auto& structureMap = output.structureMap();

    int nnz = numNonZeroes();
    indexMap.resize( nnz );
    for( int k = 0; k < nnz; ++k )
    {
        auto ij = std::make_pair( m_rowIndices[k], m_colIndices[k] );
        int l = structureMap.find( ij )->second;
        indexMap[ l ] = k;
    }
}

template< typename T >
void CoordinateSparseMatrix< T >::compressTranspose( CompressedSparseMatrix< T >& outputAt ) const
{
    int m = numRows();
    int n = numCols();
    int nnz = numNonZeroes();
    outputAt.reset( n, m, nnz );

    // copy the SparseMatrixTriplet< T > vector
    std::vector< SparseMatrixTriplet< T > > ijv( nnz );

    // flip i and j
    for( int k = 0; k < nnz; ++k )
    {
        SparseMatrixTriplet< T >& t = ijv[ k ];
        t.i = m_colIndices[ k ];
        t.j = m_rowIndices[ k ];
        t.value = m_values[ k ];
    }

    std::sort( ijv.begin(), ijv.end(), colMajorLess );

    compressCore( ijv, outputAt );

    // populate structure map
    auto& structureMap = outputAt.structureMap();
    for( int k = 0; k < nnz; ++k )
    {
        const SparseMatrixTriplet< T >& t = ijv[k];
        structureMap[ std::make_pair( t.i, t.j ) ] = k;
    }
}

template< typename T >
void CoordinateSparseMatrix< T >::compressTranspose( CompressedSparseMatrix< T >& outputAt, std::vector< int >& indexMap ) const
{
    compressTranspose( outputAt );

    const auto& structureMap = outputAt.structureMap();

    int nnz = numNonZeroes();
    indexMap.resize( nnz );
    for( int k = 0; k < nnz; ++k )
    {
        auto ji = std::make_pair( m_colIndices[k], m_rowIndices[k] );
        int l = structureMap.find( ji )->second;
        indexMap[ l ] = k;
    }
}

template<>
void CoordinateSparseMatrix< float >::multiplyVector( FloatMatrix& x, FloatMatrix& y )
{
    int m = numRows();
    int n = numCols();

    assert( x.numRows() == n );
    assert( x.numCols() == 1 );

    y.resize( m, 1 );

    // TODO: implement doubles version
    char transa = 'n';
    int nnz = numNonZeroes();

    mkl_cspblas_scoogemv
    (
        &transa,
        &m,
        m_values.data(),
        m_rowIndices.data(),
        m_colIndices.data(),
        &nnz,
        x.data(),
        y.data()
    );
}

template<>
void CoordinateSparseMatrix< float >::multiplyTransposeVector( FloatMatrix& x, FloatMatrix& y )
{
    int m = numRows();
    int n = numCols();

    assert( x.numRows() == m );
    assert( x.numCols() == 1 );

    y.resize( n, 1 );

    // TODO: implement doubles version
    char transa = 't';
    int nnz = numNonZeroes();

    mkl_cspblas_scoogemv
    (
        &transa,
        &n,
        m_values.data(),
        m_rowIndices.data(),
        m_colIndices.data(),
        &nnz,
        x.data(),
        y.data()
    );
}

//////////////////////////////////////////////////////////////////////////
// Private
//////////////////////////////////////////////////////////////////////////

template< typename T >
void CoordinateSparseMatrix< T >::compressCore( std::vector< SparseMatrixTriplet< T > > ijvSorted, CompressedSparseMatrix< T >& output ) const
{
    // TODO: if we want the output to be only symmetric or triangular
    //if( ( !upperTriangleOnly ) ||
    // ( j >= i ) )

    int nnz = numNonZeroes();
    int innerIndex = 0;
    int outerIndexPointerIndex = 0;

    auto& values = output.values();
    auto& innerIndices = output.innerIndices();
    auto& outerIndexPointers = output.outerIndexPointers();

    for( int k = 0; k < nnz; ++k )
    {
        const SparseMatrixTriplet< T >& t = ijvSorted[k];
        int i = t.i;
        int j = t.j;
        const T& value = t.value;

        values[ innerIndex ] = value;
        innerIndices[ innerIndex ] = i;

        if( j == outerIndexPointerIndex )
        {
            outerIndexPointers[ outerIndexPointerIndex ] = innerIndex;
            ++outerIndexPointerIndex;
        }
        ++innerIndex;
    }
    outerIndexPointers[ outerIndexPointerIndex ] = innerIndex;
    ++outerIndexPointerIndex;
}

// static
template< typename T >
bool CoordinateSparseMatrix< T >::rowMajorLess( SparseMatrixTriplet< T >& a, SparseMatrixTriplet< T >& b )
{
    if( a.i < b.i )
    {
        return true;
    }
    else if( a.i > b.i )
    {
        return false;
    }
    else
    {
        return( a.j < b.j );
    }
}

// static
template< typename T >
bool CoordinateSparseMatrix< T >::colMajorLess( SparseMatrixTriplet< T >& a, SparseMatrixTriplet< T >& b )
{
    if( a.j < b.j )
    {
        return true;
    }
    else if( a.j > b.j )
    {
        return false;
    }
    else
    {
        return( a.i < b.i );
    }
}

template<>
bool CoordinateSparseMatrix< float >::loadTXT( QString filename )
{
    FILE* fp = fopen( qPrintable( filename ), "r" );
    bool succeeded = ( fp != NULL );

    if( succeeded )
    {
        int m;
        int n;
        int nnz;

        fscanf( fp, "%d\t%d\t%d\n", &m, &n, &nnz );

        m_nRows = 0;
        m_nCols = 0;
        m_rowIndices.resize( nnz );
        m_colIndices.resize( nnz );
        m_values.resize( nnz );

        for( size_t k = 0; k < nnz; ++k )
        {
            int i;
            int j;
            float v;

            fscanf( fp, "%d\t%d\t%f\n", &i, &j, &v );

            m_rowIndices[ k ] = i;
            m_colIndices[ k ] = j;
            m_values[ k ] = v;

            if( i >= m_nRows )
            {
                m_nRows = i + 1;
            }
            if( j >= m_nCols )
            {
                m_nCols = j + 1;
            }
        }
        fclose( fp );
    }

    return succeeded;
}

template<>
bool CoordinateSparseMatrix< double >::loadTXT( QString filename )
{
    FILE* fp = fopen( qPrintable( filename ), "r" );
    bool succeeded = ( fp != NULL );

    if( succeeded )
    {
        int m;
        int n;
        int nnz;

        fscanf( fp, "%d\t%d\t%d\n", &m, &n, &nnz );

        m_nRows = 0;
        m_nCols = 0;
        m_rowIndices.resize( nnz );
        m_colIndices.resize( nnz );
        m_values.resize( nnz );

        for( size_t k = 0; k < nnz; ++k )
        {
            int i;
            int j;
            double v;

            fscanf( fp, "%d\t%d\t%lf\n", &i, &j, &v );

            m_rowIndices[ k ] = i;
            m_colIndices[ k ] = j;
            m_values[ k ] = v;

            if( i >= m_nRows )
            {
                m_nRows = i + 1;
            }
            if( j >= m_nCols )
            {
                m_nCols = j + 1;
            }
        }
        fclose( fp );
    }

    return succeeded;
}

template<>
bool CoordinateSparseMatrix< float >::saveTXT( QString filename )
{
    FILE* fp = fopen( qPrintable( filename ), "w" );
    bool succeeded = ( fp != NULL );

    if( succeeded )
    {
        int nnz = numNonZeroes();

        fprintf( fp, "%d\t%d\t%d\n", m_nRows, m_nCols, nnz );
        for( int k = 0; k < nnz; ++k )
        {
            fprintf( fp, "%d\t%d\t%f\n", m_rowIndices[k], m_colIndices[k], m_values[k] );
        }
        fclose( fp );
    }

    return succeeded;
}

template<>
bool CoordinateSparseMatrix< double >::saveTXT( QString filename )
{
    FILE* fp = fopen( qPrintable( filename ), "w" );
    bool succeeded = ( fp != NULL );

    if( succeeded )
    {
        int nnz = numNonZeroes();

        fprintf( fp, "%d\t%d\t%d\n", m_nRows, m_nCols, nnz );
        for( int k = 0; k < nnz; ++k )
        {
            fprintf( fp, "%d\t%d\t%lf\n", m_rowIndices[k], m_colIndices[k], m_values[k] );
        }
        fclose( fp );
    }

    return succeeded;
}

//////////////////////////////////////////////////////////////////////////
// Instantiate Templates
//////////////////////////////////////////////////////////////////////////

template
CoordinateSparseMatrix< float >;

//template
//CoordinateSparseMatrix< double >;
