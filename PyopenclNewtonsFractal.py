import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
import os
import matplotlib.pyplot as plt
import sympy as sp
import timeit
import PyToPyOpenCL#mine

"""Possible issue with input funcs where the derivative near 0 is tiny and actual func isnt ie x^n+c with large N"""

os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
def PrepNewtonsFractalGPU(fl,fprimel,dtype="cdouble_"):
    NewtonsMethod="""
    int C = 0;
    int thresh =2;
    dtype_t fval;
    dtype_t fpval;
    fval.real=100;
    dtype_t defaultval;
    defaultval.real=0;
    defaultval.imag=0;
    while (dtype_abs(fval)>precision && C<N) 
    {
        fval=__f(X[i])__;
        fpval=__fprime(X[i])__; 
        X[i]= dtype_add(X[i],dtype_neg(dtype_divide(fval,fpval)));
        C+=1;
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

def createstartvalssimp(x1,x2,y1,y2,SideLength):#this essentially creates a flattened grid for using as input into pyopenclfuncs
    Xs=np.linspace(x1,x2,SideLength,dtype=np.complex128)
    Ys=np.linspace(y1*1j,y2*1j,SideLength)
    Vals=np.zeros(SideLength*SideLength,dtype=np.complex128)
    """for i,x in enumerate(Xs):
        for j,y in enumerate(Ys):
            Vals[(i+1)*(j+1)]=x+y"""
    for i in range(SideLength):
        Vals[i*SideLength:i*SideLength+SideLength]=Xs
        Vals[i*SideLength:i*SideLength+SideLength]+=Ys[i]#every y val repeated side length times"""
    return Vals#can be reshaped into square grid    

def createstartvals(x1,x2,y1,y2,SideLength):#this essentially creates a flattened grid for using as input into pyopenclfuncs
    
        
    Xlength=int(np.ceil(abs((x1-x2)/(y1-y2)*SideLength)))
    
    Ylength=int(np.ceil((SideLength**2)/Xlength))
    
    Xs=np.linspace(-x1*1j,-x2*1j,Xlength,dtype=np.complex128)
    Ys=np.linspace(y1,y2,Ylength)
    Vals=np.zeros(Xlength*Ylength,dtype=np.complex128)
    """for i,x in enumerate(Xs):
        for j,y in enumerate(Ys):
            Vals[i,j]=x+y"""
    
    for i in range(Xlength):
        Vals[i*Ylength:i*Ylength+Ylength]=Ys
        Vals[i*Ylength:i*Ylength+Ylength]+=Xs[i]#every y val repeated side length times
    return Vals.reshape(Xlength,Ylength)
    
    

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
    meanfreq=np.mean(hist[0])
    rangestoexclude=[]
    starttime = timeit.default_timer()
    avg=0
    for i,val in enumerate(hist[0]):
        #The constant multiple is there to control how large a bucket needs to be to be chosed.
        if val<meanfreq/2:
            if i == 0:  
                rangestoexclude.append((2*hist[1][0],hist[1][0]))
            elif i==len(hist[0]):
                rangestoexclude.append((hist[1][i],2*hist[1][i]))
            else:
                rangestoexclude.append((hist[1][i],hist[1][i+1]))
        else:
            avg+=hist[1][i]/val
    print(avg)
    for ranges in rangestoexclude:
        #print(f"excluding values > {ranges[0]} or < {ranges[1]}")
        Roots[(Roots>=ranges[0])&(Roots<=ranges[1])]=avg
    print("time to overwrite anomalous values")
    print(timeit.default_timer()-starttime)
    
    #print(np.max(Roots))
    #print(np.min(Roots))
    Xlength=int(np.ceil(abs((x1-x2)/(y1-y2)*SideLength)))
    Ylength=int(np.ceil((SideLength**2)/Xlength))
    return Roots.reshape(Xlength,Ylength),extent

def WrapperOpenCltoDraw(x1,x2,y1,y2,fl,fprimel,npoints=1000, maxdepth=200,tol=1e-16):#this is so spagetified it confuses me as i write it 
    #god help me if i ever need to look at it again
    mapcl,queue=PrepNewtonsFractalGPU(fl,fprimel)
    def innerwrap(x1,x2,y1,y2):
        return NewtonsFractalPyOpenCL(x1,x2,y1,y2,npoints,mapcl,queue,tol=tol)
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
