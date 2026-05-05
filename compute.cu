// Dakshi Kaushik & Rishi Patel
// CISC 372 - Parallel Computing
// N-Body Problem - CUDA Implementation (Step 3 - Shared Memory Optimization)

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "vector.h"
#include "config.h"

#define BLOCK_SIZE 16

static vector3 *dev_Pos = NULL;
static vector3 *dev_Vel = NULL;
static double  *dev_mass = NULL;
static vector3 *dev_accels = NULL;
static int initialized = 0;

// Kernel 1: Compute pairwise accelerations using shared memory tiling
// Each thread computes accels[i][j]
// We load tiles of position and mass into shared memory to reduce global memory reads
__global__ void computeAccels(vector3 *dev_Pos, double *dev_mass, vector3 *dev_accels, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory tiles for position and mass of the j-dimension (source objects)
    __shared__ double shPos[BLOCK_SIZE][3];
    __shared__ double shMass[BLOCK_SIZE];

    // Load shared memory: each thread loads one element of the tile
    int tileJ = blockIdx.y * blockDim.y + threadIdx.y;
    if (tileJ < n) {
        shPos[threadIdx.y][0] = dev_Pos[tileJ][0];
        shPos[threadIdx.y][1] = dev_Pos[tileJ][1];
        shPos[threadIdx.y][2] = dev_Pos[tileJ][2];
        shMass[threadIdx.y] = dev_mass[tileJ];
    }
    __syncthreads();

    if (i >= n || j >= n) return;

    if (i == j) {
        dev_accels[i * n + j][0] = 0;
        dev_accels[i * n + j][1] = 0;
        dev_accels[i * n + j][2] = 0;
    } else {
        vector3 distance;
        int k;
        for (k = 0; k < 3; k++)
            distance[k] = dev_Pos[i][k] - shPos[threadIdx.y][k];

        double magnitude_sq = distance[0]*distance[0] +
                              distance[1]*distance[1] +
                              distance[2]*distance[2];
        double magnitude = sqrt(magnitude_sq);
        double accelmag = -1 * GRAV_CONSTANT * shMass[threadIdx.y] / magnitude_sq;

        dev_accels[i * n + j][0] = accelmag * distance[0] / magnitude;
        dev_accels[i * n + j][1] = accelmag * distance[1] / magnitude;
        dev_accels[i * n + j][2] = accelmag * distance[2] / magnitude;
    }
}

// Kernel 2: Sum accelerations and update positions/velocities
// Each thread handles one entity i
__global__ void updateBodies(vector3 *dev_Pos, vector3 *dev_Vel, vector3 *dev_accels, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    vector3 accel_sum = {0, 0, 0};
    int j, k;
    for (j = 0; j < n; j++) {
        for (k = 0; k < 3; k++) {
            accel_sum[k] += dev_accels[i * n + j][k];
        }
    }

    for (k = 0; k < 3; k++) {
        dev_Vel[i][k] += accel_sum[k] * INTERVAL;
        dev_Pos[i][k] += dev_Vel[i][k] * INTERVAL;
    }
}

extern "C" void compute() {
    int n = NUMENTITIES;

    if (!initialized) {
        cudaMalloc((void**)&dev_Pos,    sizeof(vector3) * n);
        cudaMalloc((void**)&dev_Vel,    sizeof(vector3) * n);
        cudaMalloc((void**)&dev_mass,   sizeof(double)  * n);
        cudaMalloc((void**)&dev_accels, sizeof(vector3) * n * n);

        cudaMemcpy(dev_mass, mass, sizeof(double)  * n, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Pos,  hPos, sizeof(vector3) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Vel,  hVel, sizeof(vector3) * n, cudaMemcpyHostToDevice);

        initialized = 1;
    }

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    computeAccels<<<gridDim, blockDim>>>(dev_Pos, dev_mass, dev_accels, n);
    cudaDeviceSynchronize();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    updateBodies<<<blocks, threads>>>(dev_Pos, dev_Vel, dev_accels, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hPos, dev_Pos, sizeof(vector3) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dev_Vel, sizeof(vector3) * n, cudaMemcpyDeviceToHost);
}
