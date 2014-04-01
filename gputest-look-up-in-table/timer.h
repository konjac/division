#include <sys/time.h>
#include <time.h>
/**
 * this file provides a way to massaure
 * time pass by microseconds
 * **/

/**
 * a timer start.start time will save in
 * `p',so don't modify it
 * **/

int start_record(timeval *p);

/**
 * end a timer.`p' will set as
 * the time during `start_record'
 * and `end_cord' called
 * **/
int end_record(timeval*);


char* time_string(timeval*);

double time_float(timeval*);
