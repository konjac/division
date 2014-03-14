#ifndef __HK__UTIL__
#define __HK__UTIL__
#include <sys/time.h>
#include <sys/unistd.h>
double getTimerValue() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double t1 = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    return t1;
}
#define malloc(x)  _mm_malloc(x, 64)
#define free(x)  _mm_free(x)
#define dprintf printf

#endif
