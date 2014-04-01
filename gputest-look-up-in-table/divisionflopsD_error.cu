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

#define BLOCK_NUM 1024
#define THREAD_NUM 512
#define ITERATION_PER_THREAD 64
#define DIV_NUM_PER_ITERATION 64


double *m;
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ double divD_fisq(double x, double number)
{
    // fisq
    union Type{
        long long i;
        double y;
    } d;
    d.y  = number;
    d.i  = 0x5fe6eb50c7aa19f9 - ( d.i >> 1 );
    d.y  = d.y * ( 1.5 - ( 0.5 * number * d.y * d.y ) );
    d.y *= d.y;

    // newton
    d.y = d.y * (2 - number * d.y);
    d.y = d.y * (2 - number * d.y);
    d.y = d.y * (2 - number * d.y);

    return x*d.y;
}

__device__ float divF_fisq(float x, float number)
{
    // fisq
    union Type{
        long i;
        float y;
    } d;
    d.y  = number;
    d.i  = 0x5f3759df - ( d.i >> 1 );
    d.y  = d.y * ( 1.5f - ( 0.5f * number * d.y * d.y ) );
    d.y *= d.y;

    // newton
    d.y = d.y * (2 - number * d.y);
    d.y = d.y * (2 - number * d.y);

    return x*d.y;
}

#define div_direct(a, b) ((a)/(b))

#define division(funcName, divF) \
__global__ void funcName(double *m) { \
    int tid = blockIdx.x * blockDim.x + threadIdx.x; \
    m[tid]=divF(tid, 20000.0); \
}

division(division_fisq, divF_fisq)
division(division_direct, div_direct)

double result_fisq[BLOCK_NUM*THREAD_NUM],
       result_direct[BLOCK_NUM*THREAD_NUM];

int main(int argc, char* argv[])
{
    gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    gpuErrchk(cudaMalloc((void**)&m, sizeof(double)*BLOCK_NUM*THREAD_NUM));

    division_fisq<<<BLOCK_NUM, THREAD_NUM>>>(m);

    gpuErrchk(cudaMemcpy(result_fisq, m, sizeof(double )*BLOCK_NUM*THREAD_NUM, cudaMemcpyDeviceToHost));

    division_direct<<<BLOCK_NUM, THREAD_NUM>>>(m);

    gpuErrchk(cudaMemcpy(result_direct, m, sizeof(double )*BLOCK_NUM*THREAD_NUM, cudaMemcpyDeviceToHost));

    double maxError=0; 
    for (int i=0;i<BLOCK_NUM*THREAD_NUM;i++)
    {
        double t=result_fisq[i]-result_direct[i];
        if (t<0) t=-t;
        if (maxError<t) maxError=t;
    }
    printf("maxError=%g\n", maxError);

    gpuErrchk(cudaFree(m));

    return 0;
}
