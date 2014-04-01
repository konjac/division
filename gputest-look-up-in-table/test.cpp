/*
 * 平方根倒数速算法(Fast Inverse Square Root)
 * http://zh.wikipedia.org/wiki/%E5%B9%B3%E6%96%B9%E6%A0%B9%E5%80%92%E6%95%B0%E9%80%9F%E7%AE%97%E6%B3%95
 *
 * 单精度
 * http://zh.wikipedia.org/wiki/%E5%96%AE%E7%B2%BE%E5%BA%A6%E6%B5%AE%E9%BB%9E%E6%95%B8
 *
 * 双精度
 * http://zh.wikipedia.org/wiki/%E9%9B%99%E7%B2%BE%E5%BA%A6%E6%B5%AE%E9%BB%9E%E6%95%B8
 */

#include <stdio.h>

inline float newtonF(float x, float b)
{
    return x*(2-b*x); 
}

inline double newtonD(double x, double b)
{
    return x*(2-b*x);
}

inline float fisqF(float number)
{
    int i;
    float y;
    y  = number;
    i  = * ( int * ) &y;
    i  = 0x5f3759df - ( i >> 1 );
    y  = * ( float * ) &i;
    y  = y * ( 1.5F - ( 0.5F * number * y * y ) );
    return y;
}

inline double fisqD(double number)
{
    long long i;
    double y;
    y  = number;
    i  = * ( long long * ) &y;
    i  = 0x5fe6eb50c7aa19f9 - ( i >> 1 );
    y  = * ( double * ) &i;
    y  = y * ( 1.5 - ( 0.5 * number * y * y ) );
    return y;
}

float divisionF(float number)
{
    float ret=fisqF(number);
    ret*=ret;
    ret=newtonF(newtonF(ret, number), number);
    return ret;
}

double divisionD(double number)
{
    double ret=fisqD(number);
    ret*=ret;
    ret=newtonD(newtonD(newtonD(ret, number), number), number);
    return ret;
}

void testF()
{
    float number, x;
    scanf("%f", &number);
    for (int i=0;i<10;i++)
    {
        if (i==0){
            x=fisqF(number);
            x=x*x;
        } else {
            x=newtonF(x, number);
        }
        printf("x[%d]=%0.20e, err=%0.20e\n ",i, x, x-1/number);
    }
    printf("newton^2=%0.20e\n", divisionF(number));
}

void testD()
{
    double number, x;
    scanf("%lf", &number);
    for (int i=0;i<10;i++)
    {
        if (i==0){
            x=fisqD(number);
            x=x*x;
        } else {
            x=newtonD(x, number);
        }
        printf("x[%d]=%0.20e, err=%0.20e\n ",i, x, x-1/number);
    }
    printf("newton^3=%0.20e\n", divisionD(number));
}

#define TABLE_SIZE (1<<12)
float table[TABLE_SIZE];
void computeTable()
{
    for (int i=0;i<TABLE_SIZE;i++)
        if (i==0) table[i]=0; else table[i]=TABLE_SIZE/2/((float)i);
}

float division_luit(float x)
{
    int n=(int)(x*TABLE_SIZE/2);
    printf("n=%d\n", n);
    float y=x*table[n]-1;
    float ret=table[n];
    printf("y=%0.20f\nret=%0.20f\n",y,ret);
    ret*=(1-y);
    printf("result=%0.20e error=%0.20e\n", ret, ret-1.0f/x);
    y*=y;
    ret*=(1+y);
    printf("result=%0.20e error=%0.20e\n", ret, ret-1.0f/x);
    y*=y;
    ret*=(1-y);
    printf("result=%0.20e error=%0.20e\n", ret, ret-1.0f/x);
    y*=y;
    ret*=(1+y);
    printf("result=%0.20e error=%0.20e\n", ret, ret-1.0f/x);
    return ret;
}

void testT()
{
    float number, x;
    scanf("%f", &number);
    float result=division_luit(number);
    printf("luit=%0.20e error=%0.20e\n", result, result-1.0f/number);
}

int main()
{
    computeTable();
    while (1)
    {
        //testD();
        //testF();
        testT();
    }
    return 0;
}
