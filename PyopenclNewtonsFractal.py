import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
import os
import matplotlib.pyplot as plt
import sympy as sp
import timeit
import Main
import PyToPyOpenCL#mine

"""Possible issue with input funcs where the derivative near 0 is tiny and actual func isnt ie x^n+c with large N"""

os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
def PrepNewtonsFractalGPU(fl,fprimel,dtype="cdouble_"):
    NewtonsMethod="""
    int C = 0;
    dtype_t fval;
    dtype_t fpval;
    fval.real=100;
    while (dtype_abs(fval)>precision && C<N) 
    {
        fval=__f(X[i])__;
        fpval=__fprime(X[i])__; 
        X[i]= dtype_add(X[i],dtype_neg(dtype_divide(fval,fpval)));
        C+=1;
    }
    
    if (C=N&&false)
    {
        
        X[i]=(dtype_t){0,0};
    }
    """
    #printf("has not converged");
    if isinstance(fl,str):
        fl=sp.lambdify(x,fl)
    if isinstance(fprimel,str):    
        fprimel=sp.lambdify(x,fp)
    flt=PyToPyOpenCL.translate(fl)
    fprimelt = PyToPyOpenCL.translate(fprimel)
    cl.tools.get_or_register_dtype(dtype+"t", np.complex128)  
    fsubbed = PyToPyOpenCL.subsfunction(flt,NewtonsMethod,"f")
    mapclstr=PyToPyOpenCL.subsfunction(fprimelt,fsubbed,"fprime")
    mapclstr=mapclstr.replace("dtype_",dtype)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    return ElementwiseKernel(ctx,dtype+"t"+"*X,int N,double precision",mapclstr,"NewtonsMethod",preamble="#define PYOPENCL_DEFINE_CDOUBLE //#include <pyopencl-complex.h>  "),queue

def createstartvals(x1,x2,y1,y2,SideLength):#this essentially creates a flattened grid for using as input into pyopenclfuncs
    Xs=np.linspace(x1,x2,SideLength,dtype=np.complex128)
    Ys=np.linspace(y1*1j,y2*1j,SideLength)
    Vals=np.zeros(SideLength*SideLength,dtype=np.complex128)
    for i in range(SideLength):
        Vals[i*SideLength:i*SideLength+SideLength]=Xs
        Vals[i*SideLength:i*SideLength+SideLength]+=Ys[i]#every y val repeated side length times
    return Vals#can be reshaped into square grid    

def NewtonsFractalPyOpenCL(x1,x2,y1,y2,SideLength,mapcl,queue,tol=1e-12,maxdepth=200,N=None,roundroots=True):
    
    #starttime = timeit.default_timer()
    extent = [x1,x2,y1,y2]
    InnitialValues=createstartvals(x1,x2,y1,y2,SideLength)
    Roots=cl.array.to_device(queue,InnitialValues) 
    mapcl(Roots,np.intc(maxdepth),np.float64(tol))
    Roots=Roots.get() 
    if roundroots:#This prevents each root from being different by less than the tol, being seen as different roots.
        #seems to result in slightly sharper image, but none insignificant speed penalty
        if N==None: N=int(np.sqrt(abs(np.log10(tol))))+1
        Roots=np.round(Roots.real+Roots.imag,N)
    else:
        Roots=Roots.real+Roots.imag
    #print(timeit.default_timer()-starttime)
    hist=np.histogram(Roots)
    np.argmax(hist[0])
    print(np.unique(Roots))
    print(len(np.unique(Roots)))
    print(np.histogram(Roots))
    plt.hist(np.histogram(Roots))
    plt.show()
    return Roots.reshape(SideLength,SideLength),extent

def WrapperOpenCltoDraw(x1,x2,y1,y2,fl,fprimel,npoints=1000, maxdepth=200,tol=1e-16):#this is so spagetified it confuses me as i write it 
    #god help me if i ever need to look at it again
    mapcl,queue=PrepNewtonsFractalGPU(fl,fprimel)
    def innerwrap(x1,x2,y1,y2):
        return NewtonsFractalPyOpenCL(x1,x2,y1,y2,npoints,mapcl,queue)
    return innerwrap

    
if __name__ == '__main__':#not really intended to be script just here for testing and demo
    x,y=sp.symbols("x,y")
    f=x**3-1j
    
    print(f)
    dtype="cdouble_"
    fp = sp.diff(f)
    fl=sp.lambdify(x,f)

    fprimel=sp.lambdify(x,fp)
    
   
    
    SideLength=1000
    
    mapcl,queue=PrepNewtonsFractalGPU(fl,fprimel)
    Roots,extent=NewtonsFractalPyOpenCL(-0.5,0.5,-0.5,0.5,1000,mapcl,queue)
    
    
    plt.imshow(Roots,extent=extent)
    plt.show()
