#include <stdio.h>
#include <util.h>
#include <immintrin.h>
#include <omp.h>

#define THREAD_NUM (61*4)
#define REPEAT (1<<15)

void division_cpu() {
    float m[THREAD_NUM] = {0};
    double t1 = getTimerValue();
    #pragma omp parallel num_threads(THREAD_NUM)
    {
        int x = omp_get_thread_num();
        for(int i = 0; i < REPEAT; i++) {
#define CAL(p, v) \
				/** float p = (i + v) * x * x * x * x /**/ \
				float p = (i + v) / x / x / x / x
            CAL(p1, 20001.0f);
            CAL(p2, 20001.1f);
            CAL(p3, 20001.2f);
            CAL(p4, 20001.3f);
            CAL(p5, 20001.4f);
            CAL(p6, 20001.5f);
            CAL(p7, 20001.6f);
            CAL(p8, 20001.7f);
            m[x] += p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8;
        }
    }
    double t2 = getTimerValue();
    //printf("%f\n", m[123]);
    printf("%-20stime = %f ms\n", __FUNCTION__, t2 - t1);
}

void division_autovec() {
    float m[THREAD_NUM] = {0};
#pragma offload target(mic) in(m:length(THREAD_NUM) alloc_if(1) free_if(0))
    {/* host memory -> mic memory */
    }
    double t1 = getTimerValue();
#pragma offload target(mic) nocopy(m)
    {

        #pragma omp parallel num_threads(THREAD_NUM)
        {
            int x = omp_get_thread_num();
            for(int i = 0; i < REPEAT; i++) {
#define CAL(p, v) \
				/** float p = (i + v) * x * x * x * x /**/ \
				float p = (i + v) / x / x / x / x
                CAL(p1, 20001.0f);
                CAL(p2, 20001.1f);
                CAL(p3, 20001.2f);
                CAL(p4, 20001.3f);
                CAL(p5, 20001.4f);
                CAL(p6, 20001.5f);
                CAL(p7, 20001.6f);
                CAL(p8, 20001.7f);
                m[x] += p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8;
            }
        }
    }
    double t2 = getTimerValue();
#pragma offload target(mic) out(m:length(THREAD_NUM) alloc_if(0) free_if(1))
    {/* free mic memory */
    }
    //printf("%f\n", m[123]);
    printf("%-20stime = %f ms\n", __FUNCTION__, t2 - t1);
}

void division_intrin() {
    float m[THREAD_NUM] = {0};
#pragma offload target(mic) in(m:length(THREAD_NUM) alloc_if(1) free_if(0))
    {/* host memory -> mic memory */
    }
    double t1 = getTimerValue();
#pragma offload target(mic) nocopy(m)
    {
        #pragma omp parallel num_threads(THREAD_NUM)
        {
            int x = omp_get_thread_num();
            for(int i = 0; i < REPEAT; i += 16) {
#ifdef __MIC__
                __m512 idv, deltav, xv, iv;
                xv     = _mm512_set1_ps(x);  // xv = [x, x, x, ..., x]
                iv     = _mm512_set1_ps(i * 1.); // iv = [i, i, i, ..., i]
                idv    = _mm512_set_ps(0., 1., 2., 3., 4., 5., 6., 7.,
                8., 9., 10., 11., 12., 13., 14., 15.); // idv = [0, 1, 2, ..., 15]
                idv    = _mm512_add_ps(iv, idv); // idv = idv + iv
#define CAL(pv, delta) __m512 pv; \
					deltav = _mm512_set1_ps(delta); \
					pv    = _mm512_add_ps(idv, deltav); \
					pv    = _mm512_div_ps(pv, xv); pv    = _mm512_div_ps(pv, xv); \
					pv    = _mm512_div_ps(pv, xv); pv    = _mm512_div_ps(pv, xv);
                CAL(pv1, 20001.0f);
                CAL(pv2, 20001.1f);
                CAL(pv3, 20001.2f);
                CAL(pv4, 20001.3f);
                CAL(pv5, 20001.4f);
                CAL(pv6, 20001.5f);
                CAL(pv7, 20001.6f);
                CAL(pv8, 20001.7f);
                pv1   = _mm512_add_ps(pv1, pv2);
                pv1   = _mm512_add_ps(pv1, pv3);
                pv1   = _mm512_add_ps(pv1, pv4);
                pv1   = _mm512_add_ps(pv1, pv5);
                pv1   = _mm512_add_ps(pv1, pv6);
                pv1   = _mm512_add_ps(pv1, pv7);
                pv1   = _mm512_add_ps(pv1, pv8);
                m[x] += _mm512_reduce_add_ps(pv1);
#endif
            }
        }
    }
    double t2 = getTimerValue();
#pragma offload target(mic) out(m:length(THREAD_NUM) alloc_if(0) free_if(1))
    {/* free mic memory */
    }
    //printf("%f\n", m[123]);
    printf("%-20stime = %f ms\n", __FUNCTION__, t2 - t1);
}
int main(int argc, char* argv[]) {
    for(int i = 0; i < 5; ++i) {
    //    division_cpu();
    //    division_cpu();
        division_intrin();
        division_intrin();
        division_autovec();
        division_autovec();
    }
}
