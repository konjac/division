#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#define N 8
#define MAGIC _mm512_set1_epi64(0x5fe6eb50c7aa19f9)
#define INT_ONES _mm512_set1_epi64(1) 
#define HALFS _mm512_set1_pd(0.5)
#define REAL_ONES _mm512_set1_pd(1.0)
#define THREEHALFS _mm512_set1_pd(1.5)
#define REAL_TWOS _mm512_set1_pd(2.0)

__attribute__((target(mic))) __m512d div_iter(__m512d a, __m512d b) {
	__m512d y = _mm512_sub_pd(b, REAL_ONES);
	__m512d c = _mm512_sub_pd(REAL_ONES, y);
	int i;
	for (i = 0; i < 5; i++) {
		y = _mm512_mul_pd(y, y);
		c = _mm512_fmadd_pd(c, y, c);
	}
	
	return _mm512_mul_pd(c, a);
}

// using fast inverse square root
// parameters: c = a / b
__attribute__((target(mic))) __m512d div_fisr(__m512d a, __m512d b) { 
	__m512i ii = _mm512_castpd_si512(b); 
	ii = _mm512_sub_epi64(MAGIC, _mm512_srlv_epi64(ii, INT_ONES)); 
	__m512d y = _mm512_castsi512_pd(ii); 
	y = _mm512_mul_pd(y, 
			_mm512_sub_pd(THREEHALFS, 
				_mm512_mul_pd(HALFS, 
					_mm512_mul_pd(b, 
						_mm512_mul_pd(y, y))))); 

	y = _mm512_mul_pd(y, _mm512_sub_pd(REAL_TWOS, _mm512_mul_pd(b, y))); 
	y = _mm512_mul_pd(y, _mm512_sub_pd(REAL_TWOS, _mm512_mul_pd(b, y))); 
	y = _mm512_mul_pd(y, _mm512_sub_pd(REAL_TWOS, _mm512_mul_pd(b, y))); 
	y = _mm512_mul_pd(y, _mm512_sub_pd(REAL_TWOS, _mm512_mul_pd(b, y))); 
	return _mm512_mul_pd(a, y); 
}


typedef union {
	__m512d vec;
	double elfms[8];
} simd_t;

int main() {
	int i;
	for (i = 0; i < N; i++) {
		// both numerator and denominator are in range (0, 1]
		double numer = (rand() + 1.0) / (RAND_MAX + 1.0);
		double denom = (rand() + 1.0) / (RAND_MAX + 1.0);
		double gold = numer / denom;

		#pragma offload target(mic) in(numer, denom, gold)
		{
			printf("--------------------------\n");
			printf("%lf / %lf =\n", numer, denom);
			printf("CPU Standard:  %lf\n", numer/denom);

			__m512d numers = _mm512_set1_pd(numer);
			__m512d denoms = _mm512_set1_pd(denom);
			simd_t result;

			result.vec = _mm512_div_pd(numers, denoms);
			printf("MIC Standard:  %lf\n", result.elfms[0]);

			result.vec = div_iter(numers, denoms);
			printf("MIC Iterative: %lf\n", result.elfms[0]);

			result.vec = div_fisr(numers, denoms);
			printf("MIC FISR:      %lf\n", result.elfms[0]);
		}
	}
}
