void fmadd_intrin() {
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
			double x = pid / 10000.0 + 1;
            m[pid] = 0.0;
#ifdef __MIC__
            __m512d xv = _mm512_set1_pd(x);
            __m512d jdv = _mm512_set_pd(0., 1., 2., 3., 4., 5., 6., 7.);
#endif
            for(int i = 0; i < REPEAT; ++i) {
#pragma unroll
                for(int j = 0; j < LOOP; j += 8) {
#ifdef __MIC__
                    __m512d jv, deltav;
                    jv    = _mm512_set1_pd(j * 1.);
                    jv    = _mm512_add_pd(jv, jdv);
#define CAL(pv, delta) __m512 pv; \
					deltav = _mm512_set1_pd(delta); \
					pv    = _mm512_fmadd_pd(jv, xv, deltav); \
					pv    = _mm512_fmadd_pd(pv, xv, deltav); \
					pv    = _mm512_fmadd_pd(pv, xv, deltav); \
					pv    = _mm512_fmadd_pd(pv, xv, deltav);

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
    printf("%-30sflops = %f Gflops\n", __FUNCTION__, flops(t2 - t1));
}
