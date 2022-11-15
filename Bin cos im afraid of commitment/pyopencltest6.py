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
for (int k=0; k<5; k++){

//printf("%i,%li,%i",X[(i*5+5*k)%25],i,5*k);
X[i+5*(k+1)]+=X[i+5*k]+1;

}

"""

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

A=np.zeros((5,5),dtype=np.int_)#1val for easy testing
A[:][0]=np.arange(5)
#A=np.arange(25).reshape((5,5))


print(A)
res_g = cl.array.to_device(queue, A)
print(res_g.ndim)
print(res_g.shape)
res_g.size=5
print(res_g.size)

mapcl = ElementwiseKernel(ctx,"int *X",mapclstr,"mapcl",preamble="#define PYOPENCL_DEFINE_CDOUBLE //#include <pyopencl-complex.h>  ")
mapcl(res_g)
print(res_g.get())
print(A)