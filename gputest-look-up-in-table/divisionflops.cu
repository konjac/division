#include "timer.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <climits>
#include "cuda.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define BLOCK_NUM 4096
#define THREAD_NUM 512
#define ITERATION_PER_THREAD 64
#define DIV_NUM_PER_ITERATION 64

#define GFLOPS(time) ((float)BLOCK_NUM*THREAD_NUM*ITERATION_PER_THREAD*DIV_NUM_PER_ITERATION/(time)/1e6)

__device__ float m[BLOCK_NUM*THREAD_NUM];
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define TABLE_BLOCK 32
#define TABLE_THREAD 32
#define TABLE_ELEMENT_PER_THREAD 8
#define TABLE_SIZE (TABLE_BLOCK*TABLE_THREAD*TABLE_ELEMENT_PER_THREAD)

__device__ float table[TABLE_SIZE];

__global__ void computeTable()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i=0;i<TABLE_ELEMENT_PER_THREAD;i++)
    {
        int j=tid*TABLE_ELEMENT_PER_THREAD+i;
        if (j==0) table[j]=0.0f; else table[j]=TABLE_SIZE/2/((float)j);
    }
}

__device__ float divF_luit(float x, float number)
{
    int n=(int)(x*TABLE_SIZE/2);
    float y=x*table[n]-1;
    float ret=table[n];
    ret*=1-y;
    y*=y;
    ret*=1+y;
    y*=y;
    ret*=1-y;
    y*=y;
    ret*=1+y;
    return x*ret;
}

__device__ float divF_fisq(float x, float number)
{
    // fisq
    union Type{
        int i;
        float y;
    } d;
    d.y  = number;
    d.i  = 0x5f3759df - ( d.i >> 1 );
    d.y  = d.y * ( 1.5F - ( 0.5F * number * d.y * d.y ) );
    d.y *= d.y;

    // newton
    d.y = d.y * (2 - number * d.y);
    d.y = d.y * (2 - number * d.y);

    return x*d.y;
}

#define divF_direct(a, b) ((a)/(b))
#define divF_fdividef(a,b) __fdividef((a), (b)) 
#define difF_add(a,b) ((a)+(b))

#define division(funcName, divF) \
__global__ void funcName() { \
    int tid = blockIdx.x * blockDim.x + threadIdx.x; \
	float x = tid; \
    for (int i = 0; i < ITERATION_PER_THREAD; i++) { \
		x += 0.0001; \
        float p1 = i + 20001.0f; \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.01f), x+0.02f), x+0.03f), x+0.04f), x+0.05f), x+0.06f), x+0.07f), x+0.08f); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.09f), x+0.10f), x+0.11f), x+0.12f), x+0.13f), x+0.14f), x+0.15f), x+0.16f); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.17f), x+0.18f), x+0.19f), x+0.20f), x+0.21f), x+0.22f), x+0.23f), x+0.24f); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.25f), x+0.26f), x+0.27f), x+0.28f), x+0.29f), x+0.30f), x+0.31f), x+0.32f); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.33f), x+0.34f), x+0.35f), x+0.36f), x+0.37f), x+0.38f), x+0.39f), x+0.40f); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.41f), x+0.42f), x+0.43f), x+0.44f), x+0.45f), x+0.46f), x+0.47f), x+0.48f); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.49f), x+0.50f), x+0.51f), x+0.52f), x+0.53f), x+0.54f), x+0.55f), x+0.56f); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.57f), x+0.58f), x+0.59f), x+0.60f), x+0.61f), x+0.62f), x+0.63f), x+0.64f); \
        m[tid] += p1; \
    } \
}

division(division_fisq, divF_fisq)
division(division_direct, divF_direct)
division(division_fdividef, divF_fdividef)
division(division_luit, divF_luit)

int main(int argc, char* argv[])
{
    gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));


    // pre-compute for table
    computeTable<<<TABLE_BLOCK, TABLE_THREAD>>>();

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time;

    cudaEventRecord( start, 0 );
    division_fisq<<<BLOCK_NUM, THREAD_NUM>>>();
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    printf("divF_fisq         time = %f ms, Gflops=%.2f \n", time, GFLOPS(time));

    cudaEventRecord( start, 0 );
    division_direct<<<BLOCK_NUM, THREAD_NUM>>>();
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    printf("divF_direct       time = %f ms, Gflops=%.2f \n", time, GFLOPS(time));

    cudaEventRecord( start, 0 );
    division_fdividef<<<BLOCK_NUM, THREAD_NUM>>>();
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    printf("divF_fdividef     time = %f ms, Gflops=%.2f \n", time, GFLOPS(time));

    cudaEventRecord( start, 0 );
    division_luit<<<BLOCK_NUM, THREAD_NUM>>>();
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    printf("divF_luit   time = %f ms, Gflops=%.2f \n", time, GFLOPS(time));

    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return 0;
}
