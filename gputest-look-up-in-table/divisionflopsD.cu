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

#define GFLOPS(time) ((float)BLOCK_NUM*THREAD_NUM*ITERATION_PER_THREAD*DIV_NUM_PER_ITERATION/(time)/1e6)

__device__ double m[BLOCK_NUM*THREAD_NUM];
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

#define USE_CONSTANT_TABLE
#ifdef USE_CONSTANT_TABLE
__device__ __constant__ double table[TABLE_SIZE];
double tableCpu[TABLE_SIZE];

void computeTable()
{
	for (int i=0;i<TABLE_SIZE;i++)
		if (i==0) tableCpu[i]=0.0; else tableCpu[i]=TABLE_SIZE/2/((double)i);
	cudaMemcpyToSymbol(table, tableCpu, sizeof(double)*TABLE_SIZE);
}
#else
__device__ double table[TABLE_SIZE];

__global__ void computeTable()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i=0;i<TABLE_ELEMENT_PER_THREAD;i++)
    {
        int j=tid*TABLE_ELEMENT_PER_THREAD+i;
        if (j==0) table[j]=0.0; else table[j]=TABLE_SIZE/2/((double)j);
    }
}
#endif

__device__ double luit(double number)
{
    int n=(int)(number*TABLE_SIZE/2);
    double y=number*table[n]-1;
    double ret=table[n];
    ret*=1-y;
    y*=y;
    ret*=1+y;
	/*
    y*=y;
    ret*=1+y;
    y*=y;
    ret*=1+y;
	*/
    return ret;
}

__device__ double divD_luit(double x, double number)
{
    union Data{
        double d;
        long long i;
    } a;
	a.d=number;
	long long e=a.i&0x7ff0000000000000;
	e=((0x7ff0000000000000-(e-0x3ff0000000000000))+0x0010000000000000);
	a.i=(a.i&0x800fffffffffffff)|0x3ff0000000000000;
    a.d=luit(a.d);
	a.i=(a.i&0x800fffffffffffff)|(((a.i&0x7ff0000000000000)+e)&0x7ff0000000000000);
	return x*a.d;
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

    return x*d.y;
}

#define divD_direct(a, b) ((a)/(b))
//#define divF_fdividef(a,b) __fdividef((a), (b)) 
#define divD_add(a,b) ((a)+(b))

#define division(funcName, divF) \
__global__ void funcName() { \
    int tid = blockIdx.x * blockDim.x + threadIdx.x; \
	double x = tid*1e-6; \
    for (int i = 0; i < ITERATION_PER_THREAD; i++) { \
		x += 0.0001; \
        double p1 = i + 20001.0; \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.01), x+0.02), x+0.03), x+0.04), x+0.05), x+0.06), x+0.07), x+0.08); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.09), x+0.10), x+0.11), x+0.12), x+0.13), x+0.14), x+0.15), x+0.16); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.17), x+0.18), x+0.19), x+0.20), x+0.21), x+0.22), x+0.23), x+0.24); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.25), x+0.26), x+0.27), x+0.28), x+0.29), x+0.30), x+0.31), x+0.32); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.33), x+0.34), x+0.35), x+0.36), x+0.37), x+0.38), x+0.39), x+0.40); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.41), x+0.42), x+0.43), x+0.44), x+0.45), x+0.46), x+0.47), x+0.48); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.49), x+0.50), x+0.51), x+0.52), x+0.53), x+0.54), x+0.55), x+0.56); \
        p1 = divF(divF(divF(divF(divF(divF(divF(divF(p1, x+0.57), x+0.58), x+0.59), x+0.60), x+0.61), x+0.62), x+0.63), x+0.64); \
        m[tid] += p1; \
    } \
}

division(division_fisq, divD_fisq)
division(division_direct, divD_direct)
division(division_add, divD_add)
division(division_luit, divD_luit)


int main(int argc, char* argv[])
{
    gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    // pre-compute for table
#ifdef USE_CONSTANT_TABLE
	computeTable();
#else
    computeTable<<<TABLE_BLOCK, TABLE_THREAD>>>();
#endif
    gpuErrchk(cudaGetLastError());

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );
    division_fisq<<<BLOCK_NUM, THREAD_NUM>>>();
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    printf("divD_fisq         time = %f ms, Gflops=%.2f \n", time, GFLOPS(time));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );
    division_direct<<<BLOCK_NUM, THREAD_NUM>>>();
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    printf("divD_direct       time = %f ms, Gflops=%.2f \n", time, GFLOPS(time));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );
    division_add<<<BLOCK_NUM, THREAD_NUM>>>();
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    printf("divD_add     time = %f ms, Gflops=%.2f \n", time, GFLOPS(time));

    cudaEventRecord( start, 0 );
    division_luit<<<BLOCK_NUM, THREAD_NUM>>>();
    gpuErrchk(cudaGetLastError());
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    gpuErrchk(cudaGetLastError());
    cudaEventElapsedTime( &time, start, stop );
    printf("divD_luit   time = %f ms, Gflops=%.2f \n", time, GFLOPS(time));

    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return 0;
}
