#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <util.h>

#define THREAD_NUM (59*4)
#define REPEAT (1<<17)
#define LOOP 1
#define NITERS 32
#define NPAR 8
#define NDIMS 8

double flops(double t) {
    return
        1.0 * NDIMS * THREAD_NUM * REPEAT * LOOP * (NPAR * NITERS) / t * 1000.0 / 1e9;
}

#define RUN1(x) x;
#define RUN2(x) x; x;
#define RUN3(x) RUN2(x) x;
#define RUN4(x) RUN2(x) RUN2(x);
#define RUN5(x) RUN4(x) x;
#define RUN6(x) RUN4(x) RUN2(x);
#define RUN7(x) RUN4(x) RUN3(x);
#define RUN8(x) RUN4(x) RUN4(x);


/*#define DIV8(a, b, c) do{ \
                        __m512 y = b; \
                        c = _mm512_fnmadd_pd(a, y, a); \
                        y = _mm512_mul_pd(y, y); \
                        c = _mm512_fmadd_pd(c, y, c);\
                        y = _mm512_mul_pd(y, y); \
                        c = _mm512_fmadd_pd(c, y, c);\
                        y = _mm512_mul_pd(y, y); \
                        c = _mm512_fmadd_pd(c, y, c);\
                        y = _mm512_mul_pd(y, y); \
                        c = _mm512_fmadd_pd(c, y, c);\
                        y = _mm512_mul_pd(y, y); \
                        c = _mm512_fmadd_pd(c, y, c);\
                    }while(0)*/

#define DIV8(a, b, c) do{ \
                        __m512 y = b; \
                        c = _mm512_fnmadd_pd(a, y, a); \
						RUN5(\
                        y = _mm512_mul_pd(y, y); \
                        c = _mm512_fmadd_pd(c, y, c);\
						); \
                    }while(0)

void fmadd_intrin() {
    double m[THREAD_NUM] = {0};
#pragma offload target(mic) in(m:length(THREAD_NUM) alloc_if(1) free_if(0))
    {}
    double t1 = getTimerValue();
#pragma offload target(mic) nocopy(m)
    {
        #pragma omp parallel num_threads(THREAD_NUM)
        {
            int pid = omp_get_thread_num();
            double x = pid / 10000.0 + 1;
            m[pid] = 0.0;
#ifdef __MIC__
            __m512d jdv = _mm512_set_pd(0., 1., 2., 3.,     4., 5., 6., 7.);
            __m512d xv = _mm512_set1_pd(x);
#endif
            for(int i = 0; i < REPEAT; ++i) {
#ifdef __MIC__
                __m512d jv[NPAR];
#pragma unroll
                for(int k = 0; k < NPAR; k++) jv[k] = _mm512_set1_pd(i);
#pragma unroll
                for(int k = 0; k < NITERS; k++) {
#pragma unroll
                    for(int l = 0; l < NPAR; l++){
						//jv[l]    = _mm512_fmadd_pd(jv[l], xv, jdv);
						//jv[l]    = _mm512_add_pd(jv[l], jdv);
						//jv[l]    = _mm512_mul_pd(jv[l], jdv);
                        //jv[l]    = _mm512_div_pd(jv[l], xv);
						DIV8(jv[l], xv, jv[l]);
						//jv[l] = _mm512_invsqrt_pd(jv[l]);
					}
                }
#pragma unroll
                for(int k = 0; k < NPAR; k++)
                    m[pid] += _mm512_reduce_add_pd(jv[k]);
#endif
            }
        }
    }
    double t2 = getTimerValue();
#pragma offload target(mic) out(m:length(THREAD_NUM) alloc_if(0) free_if(1))
    {}
    printf("%-30sflops = %f Gflops\n", __FUNCTION__, flops(t2 - t1));
}

int main(int argc, char* argv[]) {
    for(int i = 0; i < 10; ++i) fmadd_intrin();
}
