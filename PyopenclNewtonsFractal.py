import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
import os
import matplotlib.pyplot as plt
import sympy as sp
import timeit

import PyToPyOpenCL#mine

os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
def createOpenclFun(code,dtype,fl,fprimel):
    if isinstance(fl,str):
        fl=sp.lambdify(x,fl)
    if isinstance(fprimel,str):    
        fprimel=sp.lambdify(x,fp)
    flt=PyToPyOpenCL.translate(fl)
    fprimelt = PyToPyOpenCL.translate(fprimel)
    cl.tools.get_or_register_dtype("cdouble_t", np.complex128)
    fsubbed = PyToPyOpenCL.subsfunction(flt,code,"f")
    mapclstr=PyToPyOpenCL.subsfunction(fprimelt,fsubbed,"fprime")
    mapclstr=mapclstr.replace("dtype_",dtype)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    return queue,ElementwiseKernel(ctx,"cdouble_t *X,int N,double precision",mapclstr,"NewtonsMethod",preamble="#define PYOPENCL_DEFINE_CDOUBLE //#include <pyopencl-complex.h>  ")

def createstartvals(SideLength,x1,x2,y1,y2):#this essentially creates a flattened grid for using as input into pyopenclfuncs
    Xs=np.linspace(x1,x2,SideLength,dtype=np.complex128)
    Ys=np.linspace(y1*1j,y2*1j,SideLength)
    Vals=np.zeros(SideLength*SideLength,dtype=np.complex128)
    for i in range(SideLength):
        Vals[i*SideLength:i*SideLength+SideLength]=Xs
        Vals[i*SideLength:i*SideLength+SideLength]+=Ys[i]#every y val repeated side length times
    return Vals#can be reshaped into grid    
        
if __name__ == '__main__':
    x,y=sp.symbols("x,y")
    f=x**3-1
    dtype="cdouble_"
    fp = sp.diff(f)
    fl=sp.lambdify((x),f)

    fprimel=sp.lambdify(x,fp)
    flt=PyToPyOpenCL.translate(fl)
    fprimelt = PyToPyOpenCL.translate(fprimel)
    print(f)
    print(flt)
    print(fp)
    print(fprimelt)
    
   
    NewtonsMethod="""
    int C = 0;
    cdouble_t fval;
    cdouble_t fpval;
    fval.real=100;
    fpval.real=100;
    fval.imag=100;
    fpval.imag=100;
    while ((fval.real*fval.real+fval.imag*fval.imag)>precision && C<N) 
    {
        fval=__f(X[i])__;
        fpval=__fprime(X[i])__; 
        X[i]= dtype_add(X[i],dtype_neg(dtype_divide(fval,fpval)));
        C+=1;
    }"""
    SideLength=2000
    cl.tools.get_or_register_dtype("cdouble_t", np.complex128)
    fsubbed = PyToPyOpenCL.subsfunction(flt,NewtonsMethod,"f")
    print(fsubbed)
    mapclstr=PyToPyOpenCL.subsfunction(fprimelt,fsubbed,"fprime")
    mapclstr=mapclstr.replace("dtype_",dtype)
    print(mapclstr)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    #InnitialValues=np.linspace(-500,500,1000,dtype=np.complex128)#1val for easy testing
    InnitialValues=createstartvals(SideLength,-1,1,-1,1)
    Roots = cl.array.to_device(queue, InnitialValues)
    starttime = timeit.default_timer()
    mapcl = ElementwiseKernel(ctx,"cdouble_t *X,int N,double precision",mapclstr,"NewtonsMethod",preamble="#define PYOPENCL_DEFINE_CDOUBLE //#include <pyopencl-complex.h>  ")
    mapcl(Roots,np.intc(200),np.float64(1e-16))
    Roots=Roots.get() 
    Roots=np.round(abs(Roots.real+Roots.imag),5)
    Roots=Roots.reshape(SideLength,SideLength)
    print(timeit.default_timer()-starttime)
    print(len(np.unique(Roots)))
    plt.imshow(Roots)
    plt.show()
