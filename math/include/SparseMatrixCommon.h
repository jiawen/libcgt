#pragma once

#include <utility>

#include <common/BasicTypes.h>

enum MatrixTriangle
{
	LOWER,
	UPPER
};

enum MatrixType
{
	GENERAL,
	SYMMETRIC,
	TRIANGULAR
};

enum CompressedStorageFormat
{
	COMPRESSED_SPARSE_ROW,
	COMPRESSED_SPARSE_COLUMN
};

typedef std::pair< uint, uint > SparseMatrixKey;