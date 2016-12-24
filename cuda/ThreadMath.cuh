#pragma once

namespace libcgt { namespace cuda { namespace threadmath {

// ===== no unwrapping =====

// same as threadIdx, but as an int2
__inline__ __device__
int2 threadSubscript2DWithinBlock();

// same as threadIdx, but as an int3
__inline__ __device__
int3 threadSubscript3DWithinBlock();

// ==== unwrap threads within blocks =====

// returns the 1D index wrapping a 3D thread subscript (within a block)
// index = threadIdx.x + threadIdx.y * blockDim.x * + threadIdx.z * blockDim.x * blockDim.y
__inline__ __device__
int threadIndexWithinBlock();

// ===== unwrap blocks --> grids =====

// returns the 1D index wrapping a 3D block subscript (globally)
// index = blockIdx.x + blockIdx.y * gridDim.x * + blockIdx.z * gridDim.x * gridDim.y
__inline__ __device__
int blockIndexGlobal();

// ===== unwrap threads all the way to grids (globally) =====

// ----- unwrap components independently ----

// returns the 1D subscript globally across the device
// index = threadIdx + blockIdx * blockDim
// (assumes y = 0, z = 0)
__inline__ __device__
int threadSubscript1DGlobal();

// returns the 2D subscript globally across the device
// index = threadIdx + blockIdx * blockDim
// (assumes z = 0)
__inline__ __device__
int2 threadSubscript2DGlobal();

// returns the 3D subscript globally across the device
// index = threadIdx + blockIdx * blockDim
__inline__ __device__
int3 threadSubscript3DGlobal();

// ----- unwrap into single index ----

// returns the 1D index wrapping a 3D thread subscript (globally)
// first computes tsub3D = threadSubscript3DGlobal()
// then computes:
// tid = tsub3D.x + tsub3D.y * nThreadsPerRow + tsub3D.z * nThreadsPerSlice
// tid =   tsub3D.x
//       + tsub3D.y * ( blockDim.x * gridDim.x )
//       + tsub3D.z * ( blockDim.x * gridDim.x * blockDim.y * gridDim.y )
__inline__ __device__
int threadIndexGlobal();

} } } // threadmath, cuda, libcgt

#ifdef __CUDACC__

namespace libcgt { namespace cuda { namespace threadmath {

__inline__ __device__
int blockIndexGlobal()
{
	return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

__inline__ __device__
int threadIndexWithinBlock()
{
	return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

__inline__ __device__
int threadIndexGlobal()
{
	int3 tsub3D = threadSubscript3DGlobal();
	return tsub3D.x + tsub3D.y * blockDim.x * gridDim.x + tsub3D.z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
}

__inline__ __device__
int threadSubscript1DGlobal()
{
	return threadIdx.x + blockIdx.x * blockDim.x;
}

__inline__ __device__
int2 threadSubscript2DGlobal()
{
	return make_int2
	(
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y
	);
}

__inline__ __device__
int3 threadSubscript3DGlobal()
{
	return make_int3
	(
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
		threadIdx.z + blockIdx.z * blockDim.z
	);
}

__inline__ __device__
int2 threadSubscript2DWithinBlock()
{
	return make_int2( threadIdx.x, threadIdx.y );
}

__inline__ __device__
int3 threadSubscript3DWithinBlock()
{
	return make_int3( threadIdx.x, threadIdx.y, threadIdx.z );
}

} } } // threadmath, cuda, libcgt

#endif
