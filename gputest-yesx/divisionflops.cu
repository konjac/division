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

__global__ void division2() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    //m[x] = 1/x;
    m[x] = __fdividef(1.0f, x);
}

#define USE_SELF_DIV
#ifdef USE_SELF_DIV

__device__ float divF(float x, float number)
{
    // fisq
    union Type{
        int i;
        float y;
    } d;
    d.y  = number;
    d.i  = 0x5f3759df - ( d.i >> 1 );
    d.y  = d.y * ( 1.5F - ( 0.5F * number * d.y * d.y ) );

    // newton
    d.y = d.y * (2 - number * d.y);
    d.y = d.y * (2 - number * d.y);

    return x*d.y;
}

#else

//#define divF(a, b) ((a)+(b))
//#define divF(a, b) ((a)/(b))
#define divF(a, b) __fdividef((a), (b))

#endif

__global__ void division() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < ITERATION_PER_THREAD; i++) {
        /*
        float p1 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.0f), x), x), x), x), x), x), x), x);
        float p2 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.1f), x), x), x), x), x), x), x), x);
        float p3 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.2f), x), x), x), x), x), x), x), x);
        float p4 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.3f), x), x), x), x), x), x), x), x);
        float p5 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.4f), x), x), x), x), x), x), x), x);
        float p6 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.5f), x), x), x), x), x), x), x), x);
        float p7 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.6f), x), x), x), x), x), x), x), x);
        float p8 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.7f), x), x), x), x), x), x), x), x);
        */

        float p1 = i + 20001.0f;
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x), x), x), x), x), x), x), x);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x), x), x), x), x), x), x), x);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x), x), x), x), x), x), x), x);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x), x), x), x), x), x), x), x);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x), x), x), x), x), x), x), x);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x), x), x), x), x), x), x), x);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x), x), x), x), x), x), x), x);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x), x), x), x), x), x), x), x);

        /*
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.01f), x+0.02f), x+0.03f), x+0.04f), x+0.05f), x+0.06f), x+0.07f), x+0.08f);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.09f), x+0.10f), x+0.11f), x+0.12f), x+0.13f), x+0.14f), x+0.15f), x+0.16f);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.17f), x+0.18f), x+0.19f), x+0.20f), x+0.21f), x+0.22f), x+0.23f), x+0.24f);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.25f), x+0.26f), x+0.27f), x+0.28f), x+0.29f), x+0.30f), x+0.31f), x+0.32f);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.33f), x+0.34f), x+0.35f), x+0.36f), x+0.37f), x+0.38f), x+0.39f), x+0.40f);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.41f), x+0.42f), x+0.43f), x+0.44f), x+0.45f), x+0.46f), x+0.47f), x+0.48f);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.49f), x+0.50f), x+0.51f), x+0.52f), x+0.53f), x+0.54f), x+0.55f), x+0.56f);
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.57f), x+0.58f), x+0.59f), x+0.60f), x+0.61f), x+0.62f), x+0.63f), x+0.64f);
        */

        m[x] += p1;
    }
}
int main(int argc, char* argv[])
{
    gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));


    cudaEvent_t start,stop,relax_start,relax_stop,relax_non_s,relax_non_t;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&relax_start);
    cudaEventCreate(&relax_stop);
    cudaEventCreate(&relax_non_s);
    cudaEventCreate(&relax_non_t);

    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );
    division<<<BLOCK_NUM, THREAD_NUM>>>();
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    printf("time = %f ms, Mflops=%.2f \n", time, GFLOPS(time));
}
