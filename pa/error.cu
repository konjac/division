#include <stdio.h>
#include <math.h>
#include <string.h>
#include "cuda.h"

#define MAGIC 0x7fde5f73aabb2400ULL

#define BSIZE (1<<10)
#define BNUM (1<<14)

__device__ double ans[BNUM];
__global__ void division() {
	__shared__ double err[BSIZE];
	unsigned int bid = blockIdx.x;
	unsigned int bsize = blockDim.x;
	unsigned int tid = threadIdx.x;
    unsigned int idx = bid * bsize + tid;
	union{
		double d;
		unsigned int x[2];
		unsigned long long i;
	}p;
	err[tid] = 0;
	for(unsigned int i = 0; i < (1<<20); ++i){
		p.x[1] = i | 0x3ff00000;
		p.x[0] = idx << 8;
		double x = p.d;
//		p.i = MAGIC - p.i;
		p.x[0] = 0xaabb2400U - p.x[0];
		p.x[1] = 0x7fde5f73U - p.x[1];
		double r = p.d;
		double e = fabs(r*x-1);
		if(e > err[tid]) err[tid] = e;
	}
	__syncthreads();
	if(tid < 512) err[tid] = max(err[tid], err[tid+512]);
	__syncthreads();
	if(tid < 256) err[tid] = max(err[tid], err[tid+256]);
	__syncthreads();
	if(tid < 128) err[tid] = max(err[tid], err[tid+128]);
	__syncthreads();
	if(tid < 64) err[tid] = max(err[tid], err[tid+64]);
	__syncthreads();
	if(tid < 32){
		err[tid] = max(err[tid], err[tid+32]);
		err[tid] = max(err[tid], err[tid+16]);
		err[tid] = max(err[tid], err[tid+8]);
		err[tid] = max(err[tid], err[tid+4]);
		err[tid] = max(err[tid], err[tid+2]);
		err[tid] = max(err[tid], err[tid+1]);
		ans[bid] = err[0];
	}
	return;
}
int main(int argc, char* argv[]) {
	cudaEvent_t start,stop;
	float time;
    cudaEventCreate(&start), cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );
    division<<<BNUM , BSIZE>>>();
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
	double* result = (double*)malloc(sizeof(double)*BNUM);
	cudaMemcpyFromSymbol(result, ans, sizeof(double)*BNUM);
	double ans = 0.0;
	for(int i = 0; i < BNUM; ++i){
		ans = max(ans, result[i]);
	//	printf("%.20e\n", result[i]);
	}
	free(result);
	printf("total check = %lld\n", (1ULL<<20)*BSIZE*BNUM);
	printf("max error = %.20e\n", ans);
    printf("time = %g ms\n", time);
    cudaEventDestroy( start ), cudaEventDestroy( stop );
}

