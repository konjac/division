void fmadd_autovec() {
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
            for(int i = 0; i < REPEAT; i++) {
#pragma unroll
                for(int j = 0; j < LOOP; ++j) {
#define CAL(p, v) \
				/** double p = (j + v) * x * x * x * x /**/ \
				double p = (j * x + v); \
				p = p * x + v; \
				p = p * x + v; \
				p = p * x + v; 

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
    printf("%-30sflops = %f Gflops\n", __FUNCTION__, flops(t2 - t1));
}
