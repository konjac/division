#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <util.h>

#define dprintf //

#define THREAD_NUM (60*4)
#define REPEAT (1<<16)
#define LOOP 64
#define checkpos 123

double flops(double t) {
    return 1.0 * THREAD_NUM * REPEAT * LOOP * (4 * 8) / t * 1000.0 / 1e9;
}

#include "division-cpu.h"
#include "division-autovec.h"
#include "division-intrin.h"
#include "fmadd-autovec.h"
#include "fmadd-intrin.h"
#include "newdiv-autovec.h"
#include "newdiv-intrin.h"

int main(int argc, char* argv[]) {
		printf("**************** Test for FMADD ****************\n");
        fmadd_intrin();
        fmadd_intrin();
        fmadd_autovec();
        fmadd_autovec();

		printf("****************  Test for DIV  ****************\n");
        division_cpu();
        division_cpu();

		division_intrin();
        division_intrin();

        division_autovec();
        division_autovec();

        newdiv_autovec();
        newdiv_autovec();

        newdiv_intrin();
        newdiv_intrin();

}
