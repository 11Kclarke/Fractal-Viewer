import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
import os
import matplotlib.pyplot as plt
import sympy as sp
import timeit
import PyToPyOpenCL#mine
from PyopenclNewtonsFractal import createstartvals#mine

os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

StabilityFractalNoCycle="""
    int n = 0;
    dtype_t Const=X[i];//dont rename
    bool notfound = true;
    while (dtype_abs(X[i])<DivLim && n<N) 
    {
        n+=1;
        X[i]=__f(X[i],Const)__;
    }
    X[i].real=__colouring_mode__;
    """
#Gospers algorithm used as cycle detection.
#implementation of gospers algorithm modified from hackers delight. 
StabilityFractalWithCycle="""
short k,kmax,m,n;
bool notfound = true;  
bool notdiverged = true;
dtype_t T[maxCyclelog2+1];
dtype_t Xn=X[i];
dtype_t Const = Xn;
T[0] = Xn;
X[i].real = 0;
for (n = 1; n<N; n++) {
    Xn =__f(Xn,Const)__;
    kmax = maxCyclelog2-1 - clz(n); // Floor(log2 n).
    for (k = 0; k <= kmax; k++) {
        if (dtype_abs(dtype_add(Xn,dtype_neg(T[k])))<cycleacc){
            // Compute m = max{i | i < n and ctz(i+1) = k}.
            m = ((((n >> k) - 1) | 1) << k) - 1;
            X[i].real =m-n;
            notfound=false;
            n=N+1;
        } 
    }
    T[popcount(~((n+1) | -(n+1)))] = Xn; // No match.
    if (dtype_abs_squared(Xn)>DivLim){
        X[i].real=__colouring_mode__;
        notdiverged = false;
        n=N+2;
    }   
}
    """

"""Fl seed func, cycles false or 0 for no cycle detection int for number of cycles"""
def PrepStabilityFractalGPU(fl,dtype="cdouble_",cycles=False,ittCountColouring=False):
    if isinstance(fl,str):
        fl=sp.lambdify(x,fl)
    flt=PyToPyOpenCL.translate(fl)
    cl.tools.get_or_register_dtype(dtype+"t", np.complex128)
    if cycles==False or cycles==0:#not sure about behaviour of not cycles when cycles not a bool
        Code=StabilityFractalNoCycle    
    elif  isinstance(cycles,int):
        Code=StabilityFractalWithCycle
    if ittCountColouring:   
        Code= Code.replace("__colouring_mode__;","X[i].real = n;")
    else:
        Code= Code.replace("__colouring_mode__;","X[i].real = Xn.real;")#Comments the code for colouring by ittcount   
    if isinstance(cycles,str):
        Code=cycles
    
    
    mapclstr = PyToPyOpenCL.subsfunction(flt,Code,"f")
    print(mapclstr)                                                                                                          
    mapclstr=mapclstr.replace("dtype_",dtype)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    if cycles==False or cycles==0:
        return ElementwiseKernel(ctx,dtype+"t"+" *X,int N,double DivLim",mapclstr,"StabilityFractal",preamble="#define PYOPENCL_DEFINE_CDOUBLE //#include <pyopencl-complex.h>  "),queue
    else:
        preamble="""
        #define PYOPENCL_DEFINE_CDOUBLE 
        #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
        #include <pyopencl-complex.h>
        
        __constant int maxCyclelog2 = """+str(cycles)+"""; // should be set to 32 nearly always
        """
        return ElementwiseKernel(ctx,dtype+"t"+" *X,int N,double DivLim,double cycleacc",mapclstr,"StabilityFractal",preamble=preamble),queue
    

def StabilityFractalPyOpenCL(x1,x2,y1,y2,SideLength,mapcl,queue,DivLim=2.0,maxdepth=30,cycles=32,cycleacc=None,shuffle=False):
    DivLim=DivLim**2
    extent = [x1,x2,y1,y2]
    origTime=timeit.default_timer()
    Stabilities=createstartvals(x1,x2,y1,y2,SideLength)
    print(f"Time To create startvals: {timeit.default_timer()-origTime}")
    Stabilities=Stabilities.flatten()
    
    if shuffle:
        Time=timeit.default_timer()
        orderTemplate = np.arange(SideLength**2)
        shuf_order = orderTemplate
        np.random.shuffle(shuf_order)
        Stabilities=Stabilities[shuf_order]
        shuffletime = timeit.default_timer()-Time
        
        #print(shuffletime)
    Stabilities=cl.array.to_device(queue,Stabilities)

    Xlength=int(np.ceil(abs((x1-x2)/(y1-y2)*SideLength)))
    Ylength=int(np.ceil((SideLength**2)/Xlength))
    
    if cycles==False or int(cycles) == 0:#not sure about behaviour of not cycles when cycles not a bool
        Time=timeit.default_timer()
        
        mapcl(Stabilities,maxdepth,np.float64(DivLim))
    else:
        if cycleacc == None:
            cycleacc=np.sqrt((((x1-x2)**2+(y1-y2)**2)/(Xlength**2+Ylength**2)))
            #default threshold is 50% of diagonal length of 1 pixel 
            #print(f"cycle tolerance auto calced as {cycleacc}")
        Time=timeit.default_timer()
        mapcl(Stabilities,maxdepth,np.float64(DivLim),np.float64(cycleacc))
        
    Stabilities=np.real(Stabilities.get())

    print(f"Time OpenCl: {timeit.default_timer()-Time}")
    if shuffle:
        Time=timeit.default_timer()
        unshuf_order = np.zeros_like(orderTemplate)
        unshuf_order[shuf_order] = np.arange(SideLength**2)  
        Stabilities=Stabilities[unshuf_order]
        unshuffletime = timeit.default_timer()-Time
        print(unshuffletime)
        print("total shuffle unshuffle time:")
        print(shuffletime+unshuffletime)
    print(np.min(Stabilities))
    print(np.max(Stabilities))
    #print(maxdepth)

    print(f"whole generatorfucntion time {timeit.default_timer()-origTime}")
    return Stabilities.reshape(Xlength,Ylength),extent

def WrapperOpenCltoDraw(x1,x2,y1,y2,fl,npoints=1000,Divlim=2.0,maxdepth=30,dtype="cdouble_"
                    ,Code=StabilityFractalNoCycle,cycles=False,cycleacc=1e-5,ittCountColouring=True):
    openclfunc,queue= PrepStabilityFractalGPU(fl,dtype=dtype,cycles=cycles,ittCountColouring=ittCountColouring)
    def innerwrap(x1,x2,y1,y2):
        return StabilityFractalPyOpenCL(x1,x2,y1,y2,npoints,openclfunc,queue,DivLim=Divlim,maxdepth=maxdepth,cycles=cycles,cycleacc=cycleacc)
    return innerwrap

if __name__ == '__main__':#not really intended to be script just here for testing and demo
    x,Const=sp.symbols("x,Const")
    f=x**2+Const
    fl=sp.lambdify((x,Const),f)


   
    
    SideLength=500
    
    mapcl,queue=PrepStabilityFractalGPU(fl,cycles=16,ittCountColouring=True)
    
    starttime = timeit.default_timer()
    Roots,extent=StabilityFractalPyOpenCL(1,-1,1,-1,512,mapcl,queue,maxdepth=2000,cycles=16,shuffle=True)
    print(timeit.default_timer()-starttime)
    
    plt.imshow(Roots,extent=extent)
    plt.show(block=True)