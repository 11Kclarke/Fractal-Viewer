import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
import os

os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

"""
  fval=_add(_pow( X[i] , 3 ) + _pow( X[i] , 2 ) + X[i],- 1.0);
  fpval=_add(_mul(3,_pow( X[i] , 2 ))  + _mul(2,X[i]),1);
  X[i] =_add(X[i],-_divide(fval,fpval));
"""
mapclstr="""
int C = 0;
cdouble_t fval;
cdouble_t fpval;
fval.real=100;
fpval.real=100;
fval.imag=100;
fpval.imag=0;

while ((fval.real*fval.real+fval.imag*fval.imag)>precision && C<N) 
{
    
  fval=cdouble_pow(fval,(1,1j));
  C+=1;
} 
"""

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

A=np.linspace(3,6,1,dtype=np.complex128)#1val for easy testing
res_g = cl.array.to_device(queue, A)
mapcl = ElementwiseKernel(ctx,"cdouble_t *X,int N,double precision",mapclstr,"mapcl",preamble="#define PYOPENCL_DEFINE_CDOUBLE //#include <pyopencl-complex.h>  ")
mapcl(res_g,np.intc(500),np.float64(0.00001)) 
print(res_g.get())  