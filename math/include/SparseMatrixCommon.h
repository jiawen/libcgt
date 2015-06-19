#pragma once

#include <cstdint>
#include <map>
#include <unordered_map>
#include <utility>

#include "MatrixCommon.h"

typedef std::pair< uint32_t, uint32_t > SparseMatrixKey;

template< typename T >
struct SparseMatrixTriplet
{
	uint32_t i;
	uint32_t j;
	T value;
};

// compare j first, then i
struct SparseMatrixKeyColMajorLess
{
	bool operator () ( const SparseMatrixKey& a, const SparseMatrixKey& b ) const;
};

struct SparseMatrixKeyHash
{
	size_t operator () ( const SparseMatrixKey& x ) const;
};


typedef std::map< SparseMatrixKey, uint32_t > SparseMatrixStructureTreeMap;
typedef std::unordered_map< SparseMatrixKey, uint32_t, SparseMatrixKeyHash > SparseMatrixStructureHashMap;
