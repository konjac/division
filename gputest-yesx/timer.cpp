#include <stdio.h>
#include "timer.h"

int start_record(timeval *p)
{
    int result = gettimeofday(p,NULL);
    if (result == 0)
        return 1;
    else return -1;
}

int end_record(timeval *p)
{
    struct timeval tmp;
    int result = gettimeofday(&tmp,NULL);
    /* fprintf(stderr,"get time %zu sec %zu usec\n",tmp.tv_sec,tmp.tv_usec); */
    if (result == -1)
        return 0;
    p->tv_sec = tmp.tv_sec - p->tv_sec;
    if (p->tv_usec < tmp.tv_usec)
        p->tv_usec = tmp.tv_usec - p->tv_usec;
    else {
        p->tv_sec--;
        p->tv_usec = 1000000 - p->tv_usec + tmp.tv_usec;
    }
    /* fprintf(stderr,"run time %zu sec %zu usec\n",p->tv_sec,p->tv_usec); */
    return 1;
}


char* time_string(timeval *p) {
    static char result[100];
    sprintf(result,"%zu.%06zu",p->tv_sec,p->tv_usec);
    return result;
}

double time_float(timeval *p) {
    return p->tv_sec + p->tv_usec / 1e6;
}
