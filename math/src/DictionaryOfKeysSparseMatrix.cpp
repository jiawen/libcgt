#include "DictionaryOfKeysSparseMatrix.h"

#include <cassert>

#include "CompressedSparseMatrix.h"

template< typename T >
uint32_t DictionaryOfKeysSparseMatrix< T >::numRows() const
{
    return m_nRows;
}

template< typename T >
uint32_t DictionaryOfKeysSparseMatrix< T >::numCols() const
{
    return m_nCols;
}

template< typename T >
uint32_t DictionaryOfKeysSparseMatrix< T >::numNonZeroes() const
{
    return static_cast< uint32_t >( m_values.size() );
}

template< typename T >
T DictionaryOfKeysSparseMatrix< T >::operator () ( uint32_t i, uint32_t j ) const
{
    SparseMatrixKey key( i, j );
    auto itr = m_values.find( key );

    // find returns end() when it doesn't find anything
    if( itr != m_values.end() )
    {
        return itr->second;
    }
    else
    {
        return 0;
    }
}

template< typename T >
void DictionaryOfKeysSparseMatrix< T >::put( uint32_t i, uint32_t j, T value )
{
    if( value != 0 )
    {
        if( i >= m_nRows )
        {
            m_nRows = i + 1;
        }
        if( j >= m_nCols )
        {
            m_nCols = j + 1;
        }

        m_values[ SparseMatrixKey( i, j ) ] = value;
    }
}

template< typename T >
void DictionaryOfKeysSparseMatrix< T >::compress( CompressedSparseMatrix< T >& output,
    bool oneBased, bool upperTriangleOnly ) const
{
    int nnz = numNonZeroes();
    uint32_t m = numRows();
    uint32_t n = numCols();

    output.reset( m, n, nnz );

    uint32_t columnsIndex = 0;
    uint32_t rowIndexIndex = 0;
    uint32_t offset = oneBased ? 1 : 0;

    // if column compressed
    // then build a vector of (i,j,v) and sort into (j,i)

    auto& values = output.values();
    auto& innerIndices = output.innerIndices();
    auto& outerIndexPointers = output.outerIndexPointers();
    auto& structureMap = output.structureMap();

    for( auto itr = m_values.begin(); itr != m_values.end(); ++itr )
    {
        auto ij = itr->first;
        uint32_t i = ij.first;
        uint32_t j = ij.second;

        if( ( !upperTriangleOnly ) ||
            ( j >= i ) )
        {
            T value = itr->second;

            values[ columnsIndex ] = value;
            innerIndices[ columnsIndex ] = j + offset;

            structureMap[ ij ] = columnsIndex;

            if( i == rowIndexIndex )
            {
                outerIndexPointers[ rowIndexIndex ] = columnsIndex;
                ++rowIndexIndex;
            }

            ++columnsIndex;
        }
    }
    outerIndexPointers[ rowIndexIndex ] = columnsIndex + offset;
    ++rowIndexIndex;
}

template< typename T >
QString DictionaryOfKeysSparseMatrix< T >::toString() const
{
    // TODO: implement me
    return "";
}

//////////////////////////////////////////////////////////////////////////
// Instantiate Templates
//////////////////////////////////////////////////////////////////////////

template
DictionaryOfKeysSparseMatrix< float >;

//template
//DictionaryOfKeysSparseMatrix< double >;