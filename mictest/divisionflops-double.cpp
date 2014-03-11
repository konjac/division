#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <util.h>

#define THREAD_NUM (60*4)
#define REPEAT (1<<16)
#define LOOP 64
#define checkpos 123

double flops(double t) {
    return 1.0 * THREAD_NUM * REPEAT * LOOP * (4 * 8) / t * 1000.0 / (1 << 30);
}

void division_cpu() {
    double m[THREAD_NUM] = {0};
    double t1 = getTimerValue();
    #pragma omp parallel num_threads(THREAD_NUM)
    {
        int pid = omp_get_thread_num();
		double x = pid;
		m[pid] = 0.0;
        for(int i = 0; i < REPEAT; i++) {
#pragma unroll
            for(int j = 0; j < LOOP; ++j) {
#define CAL(p, v) \
				/** double p = (i + v) * x * x * x * x /**/ \
				double p = (j + v) / x / x / x / x
                CAL(p1, 20001.0f);
                CAL(p2, 20001.1f);
                CAL(p3, 20001.2f);
                CAL(p4, 20001.3f);
                CAL(p5, 20001.4f);
                CAL(p6, 20001.5f);
                CAL(p7, 20001.6f);
                CAL(p8, 20001.7f);
                m[pid] += p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8;
            }
        }
    }
    double t2 = getTimerValue();
    dprintf("%.20f\n", m[checkpos]);
    printf("%-20stime = %f Gflops\n", __FUNCTION__, flops(t2 - t1));
}

void division_autovec() {
    double m[THREAD_NUM] = {0};
#pragma offload target(mic) in(m:length(THREAD_NUM) alloc_if(1) free_if(0))
    {
        /* host memory -> mic memory */
    }
    double t1 = getTimerValue();
#pragma offload target(mic) nocopy(m)
    {

        #pragma omp parallel num_threads(THREAD_NUM)
        {
            int pid = omp_get_thread_num();
			double x = pid;
            m[pid] = 0.0;
            for(int i = 0; i < REPEAT; i++) {
#pragma unroll
                for(int j = 0; j < LOOP; ++j) {
#define CAL(p, v) \
				/** double p = (j + v) * x * x * x * x /**/ \
				double p = (j + v) / x / x / x / x

                    CAL(p1, 20001.0f);
                    __asm("#>>>>>>>>>>>>>>");
                    CAL(p2, 20001.1f);
                    CAL(p3, 20001.2f);
                    CAL(p4, 20001.3f);
                    CAL(p5, 20001.4f);
                    CAL(p6, 20001.5f);
                    CAL(p7, 20001.6f);
                    CAL(p8, 20001.7f);
                    m[pid] += p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8;
                }
            }
        }
    }
    double t2 = getTimerValue();
#pragma offload target(mic) out(m:length(THREAD_NUM) alloc_if(0) free_if(1))
    {
        /* free mic memory */
    }
    dprintf("%.20f\n", m[checkpos]);
    printf("%-20stime = %f Gflops\n", __FUNCTION__, flops(t2 - t1));
}

void division_intrin() {
    double m[THREAD_NUM] = {0};
#pragma offload target(mic) in(m:length(THREAD_NUM) alloc_if(1) free_if(0))
    {
        /* host memory -> mic memory */
    }
    double t1 = getTimerValue();
#pragma offload target(mic) nocopy(m)
    {
        #pragma omp parallel num_threads(THREAD_NUM)
        {
            int pid = omp_get_thread_num();
			double x = pid;
            m[pid] = 0.0;
#ifdef __MIC__
            __m512d xv = _mm512_set1_pd(x);
            __m512d iv = _mm512_set_pd(0., 1., 2., 3., 4., 5., 6., 7.);
#endif
            for(int i = 0; i < REPEAT; ++i) {
#pragma unroll
                for(int j = 0; j < LOOP; j += 8) {
#ifdef __MIC__
                    __m512d idv, deltav;
                    idv    = _mm512_set1_pd(j * 1.);
                    idv    = _mm512_add_pd(iv, idv);
#define CAL(pv, delta) __m512 pv; \
					deltav = _mm512_set1_pd(delta); \
					pv    = _mm512_add_pd(idv, deltav); \
					pv    = _mm512_div_pd(pv, xv); pv    = _mm512_div_pd(pv, xv); \
					pv    = _mm512_div_pd(pv, xv); pv    = _mm512_div_pd(pv, xv);
                    CAL(pv1, 20001.0f);
                    CAL(pv2, 20001.1f);
                    CAL(pv3, 20001.2f);
                    CAL(pv4, 20001.3f);
                    CAL(pv5, 20001.4f);
                    CAL(pv6, 20001.5f);
                    CAL(pv7, 20001.6f);
                    CAL(pv8, 20001.7f);
                    pv1   = _mm512_add_pd(pv1, pv5);
                    pv2   = _mm512_add_pd(pv2, pv6);
                    pv3   = _mm512_add_pd(pv3, pv7);
                    pv4   = _mm512_add_pd(pv4, pv8);
                    pv1   = _mm512_add_pd(pv1, pv3);
                    pv2   = _mm512_add_pd(pv2, pv4);
                    pv1   = _mm512_add_pd(pv1, pv2);
                    m[pid] += _mm512_reduce_add_pd(pv1);
#endif
                }
            }
        }
    }
    double t2 = getTimerValue();
#pragma offload target(mic) out(m:length(THREAD_NUM) alloc_if(0) free_if(1))
    {
        /* free mic memory */
    }
    dprintf("%.20f\n", m[checkpos]);
    printf("%-20stime = %f Gflops\n", __FUNCTION__, flops(t2 - t1));
}
int main(int argc, char* argv[]) {
    for(int i = 0; i < 2; ++i) {
        division_cpu();
        division_intrin();
        division_autovec();
        division_cpu();
        division_intrin();
        division_autovec();
    }
}
