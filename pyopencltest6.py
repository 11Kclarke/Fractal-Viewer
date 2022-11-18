import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
import os
from PyToPyOpenCL import *
import sympy as sp
import matplotlib.pyplot as plt
os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

"""
  fval=_add(_pow( X[i] , 3 ) + _pow( X[i] , 2 ) + X[i],- 1.0);
  fpval=_add(_mul(3,_pow( X[i] , 2 ))  + _mul(2,X[i]),1);
  X[i] =_add(X[i],-_divide(fval,fpval));
"""
mapclstr="""



X[i]=__f(X[i])__



"""

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

A=np.linspace(0,np.pi*2+1j,1000,dtype=np.complex128)#1val for easy testing
x,c=sp.symbols("x,c")
f=x**2
fl=sp.lambdify(x,f)
flt=translate(fl)
mapclstr = subsfunction(flt,mapclstr,"f")
#A=np.arange(25).reshape((5,5))


print(A)
res_g = cl.array.to_device(queue, A)


mapcl = ElementwiseKernel(ctx,"int *X",mapclstr,"mapcl",preamble="#define PYOPENCL_DEFINE_CDOUBLE //#include <pyopencl-complex.h>  ")
mapcl(res_g)
#print(res_g.get())
plt.plot(res_g.get())
plt.show()