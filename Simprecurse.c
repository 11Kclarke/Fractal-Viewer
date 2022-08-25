#include <complex.h>
#include <stdio.h>
  
int main(void)
{
    double real = 1.3,
           imag = 4.9;
    double complex z
        = CMPLX(real, imag);
    printf("Absolute value = %.1f",
           cabsf(z));
}
