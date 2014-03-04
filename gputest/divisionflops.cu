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

__device__ float m[4096*256];
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

__global__ void division() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 1024; i++) {
        /* float p1 = (i+20001.0f)*x*x; */
        /* float p2 = (i+20001.1f)*x*x; */
        /* float p3 = (i+20001.2f)*x*x; */
        /* float p4 = (i+20001.3f)*x*x; */
        /* float p5 = (i+20001.4f)*x*x; */
        /* float p6 = (i+20001.5f)*x*x; */
        /* float p7 = (i+20001.6f)*x*x; */
        /* float p8 = (i+20001.7f)*x*x; */

        /* float p1 = (i+20001.0f)*x*x*x*x; */
        /* float p2 = (i+20001.1f)*x*x*x*x; */
        /* float p3 = (i+20001.2f)*x*x*x*x; */
        /* float p4 = (i+20001.3f)*x*x*x*x; */
        /* float p5 = (i+20001.4f)*x*x*x*x; */
        /* float p6 = (i+20001.5f)*x*x*x*x; */
        /* float p7 = (i+20001.6f)*x*x*x*x; */
        /* float p8 = (i+20001.7f)*x*x*x*x; */

        /* float p1 = (i+20001.0f)*x*x*x*x*x*x*x*x; */
        /* float p2 = (i+20001.1f)*x*x*x*x*x*x*x*x; */
        /* float p3 = (i+20001.2f)*x*x*x*x*x*x*x*x; */
        /* float p4 = (i+20001.3f)*x*x*x*x*x*x*x*x; */
        /* float p5 = (i+20001.4f)*x*x*x*x*x*x*x*x; */
        /* float p6 = (i+20001.5f)*x*x*x*x*x*x*x*x; */
        /* float p7 = (i+20001.6f)*x*x*x*x*x*x*x*x; */
        /* float p8 = (i+20001.7f)*x*x*x*x*x*x*x*x; */

        /* float p1 = (i+20001.0f)/x/x/x/x/x/x/x/x; */
        /* float p2 = (i+20001.1f)/x/x/x/x/x/x/x/x; */
        /* float p3 = (i+20001.2f)/x/x/x/x/x/x/x/x; */
        /* float p4 = (i+20001.3f)/x/x/x/x/x/x/x/x; */
        /* float p5 = (i+20001.4f)/x/x/x/x/x/x/x/x; */
        /* float p6 = (i+20001.5f)/x/x/x/x/x/x/x/x; */
        /* float p7 = (i+20001.6f)/x/x/x/x/x/x/x/x; */
        /* float p8 = (i+20001.7f)/x/x/x/x/x/x/x/x; */
/*
         float p1 = (i+20001.0f)/x/x/x/x;
         float p2 = (i+20001.1f)/x/x/x/x;
         float p3 = (i+20001.2f)/x/x/x/x;
         float p4 = (i+20001.3f)/x/x/x/x;
         float p5 = (i+20001.4f)/x/x/x/x;
         float p6 = (i+20001.5f)/x/x/x/x;
         float p7 = (i+20001.6f)/x/x/x/x;
         float p8 = (i+20001.7f)/x/x/x/x;
*/

        float p1 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.0f), x), x), x), x), x), x), x), x);
        float p2 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.1f), x), x), x), x), x), x), x), x);
        float p3 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.2f), x), x), x), x), x), x), x), x);
        float p4 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.3f), x), x), x), x), x), x), x), x);
        float p5 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.4f), x), x), x), x), x), x), x), x);
        float p6 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.5f), x), x), x), x), x), x), x), x);
        float p7 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.6f), x), x), x), x), x), x), x), x);
        float p8 = __fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef(__fdividef((i+20001.7f), x), x), x), x), x), x), x), x);

        /* float p1 = __fdividef(__fdividef(__fdividef(__fdividef((i+20001.0f), x), x), x), x); */
        /* float p2 = __fdividef(__fdividef(__fdividef(__fdividef((i+20001.1f), x), x), x), x); */
        /* float p3 = __fdividef(__fdividef(__fdividef(__fdividef((i+20001.2f), x), x), x), x); */
        /* float p4 = __fdividef(__fdividef(__fdividef(__fdividef((i+20001.3f), x), x), x), x); */
        /* float p5 = __fdividef(__fdividef(__fdividef(__fdividef((i+20001.4f), x), x), x), x); */
        /* float p6 = __fdividef(__fdividef(__fdividef(__fdividef((i+20001.5f), x), x), x), x); */
        /* float p7 = __fdividef(__fdividef(__fdividef(__fdividef((i+20001.6f), x), x), x), x); */
        /* float p8 = __fdividef(__fdividef(__fdividef(__fdividef((i+20001.7f), x), x), x), x); */

        /* float p1 = (i+1.0f)/x/x; */
        /* float p2 = (i+1.1f)/x/x; */
        /* float p3 = (i+1.2f)/x/x; */
        /* float p4 = (i+1.3f)/x/x; */
        /* float p5 = (i+1.4f)/x/x; */
        /* float p6 = (i+1.5f)/x/x; */
        /* float p7 = (i+1.6f)/x/x; */
        /* float p8 = (i+1.7f)/x/x; */
        m[x] += p1+p2+p3+p4+p5+p6+p7+p8;
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
    division<<<4096 , 256>>>();
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    printf("time = %f ms\n", time);
}
