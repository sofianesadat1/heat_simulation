#include "headers.h"

int __host__ __device__ getIndex(const int i, const int j, const int width){
    return i*width + j;
}

__global__ void evolve_kernel(const float* Un, float* Unp1, const int nx, const int ny, const float dx2, const float dy2, const float dt, const float a){
    __shared__ float s_Un[(BLOCK_SIZE_X + 2)*(BLOCK_SIZE_Y + 2)];
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;

    int s_i = threadIdx.x + 1;
    int s_j = threadIdx.y + 1;
    int s_ny = BLOCK_SIZE_Y + 2;

    // Load data into shared memory
    // Central 
    s_Un[getIndex(s_i, s_j, s_ny)] = Un[getIndex(i, j, ny)];
    // Top border
    if (s_i == 1 && i != 0)
    {
        s_Un[getIndex(0, s_j, s_ny)] = Un[getIndex(blockIdx.x*blockDim.x - 1, j, ny)];
    }
    // Bottom border
    if (s_i == BLOCK_SIZE_X && i != nx - 1)
    {
        s_Un[getIndex(BLOCK_SIZE_X + 1, s_j, s_ny)] = Un[getIndex((blockIdx.x + 1)*blockDim.x, j, ny)];
    }
    // Left border
    if (s_j == 1 && j != 0)
    {
        s_Un[getIndex(s_i, 0, s_ny)] = Un[getIndex(i, blockIdx.y*blockDim.y - 1, ny)];
    }
    // Right border
    if (s_j == BLOCK_SIZE_Y && j != ny - 1)
    {
        s_Un[getIndex(s_i, BLOCK_SIZE_Y + 1, s_ny)] = Un[getIndex(i, (blockIdx.y + 1)*blockDim.y, ny)];
    }

    __syncthreads();

    if (i > 0 && i < nx - 1)
    {
        if (j > 0 && j < ny - 1)
        {
            float uij = s_Un[getIndex(s_i, s_j, s_ny)];
            float uim1j = s_Un[getIndex(s_i-1, s_j, s_ny)];
            float uijm1 = s_Un[getIndex(s_i, s_j-1, s_ny)];
            float uip1j = s_Un[getIndex(s_i+1, s_j, s_ny)];
            float uijp1 = s_Un[getIndex(s_i, s_j+1, s_ny)];

            Unp1[getIndex(i, j, ny)] = uij + a * dt * ( (uim1j - 2.0*uij + uip1j)/dx2 + (uijm1 - 2.0*uij + uijp1)/dy2 );
        }
    }
}

void launchEvolveKernel(float* d_Un, float* d_Unp1, int nx, int ny, float dx2, float dy2, float dt, float a) {
    dim3 numBlocks((nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_Un, d_Unp1, nx, ny, dx2, dy2, dt, a);
}
