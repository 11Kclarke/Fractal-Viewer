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
from numba import njit
os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

Burningshipvariation="Xn=(dtype_t){fabs(Xn.real),fabs(Xn.imag)};"
Tricornpvariation="Xn=dtype_conj(Xn);"

StabilityFractalNoCycle="""
    int n = 0;
    dtype_t Const=X[i];//dont rename
    bool notfound = true;
    while (dtype_abs(X[i])<DivLim && n<N) 
    {
        Xn =__variation_mode__;
        n+=1;
        X[i]=__f(X[i],Const)__;
    }
    X[i].real=__colouring_mode__;
    """
#Gospers algorithm used as cycle detection.
#implementation of gospers algorithm modified from hackers delight. 
StabilityFractalWithCycle="""
short k,kmax,m,n;
dtype_t T[maxCyclelog2+1];
dtype_t Xn=X[i];//should prolly be 0
dtype_t Const = Xn;
T[0] = Xn;
for (n = 1; n<N; n++) {
    __variation_mode__;
    Xn =__f(Xn,Const)__;
    kmax = maxCyclelog2-1 - clz(n); // Floor(log2 n).
    for (k = 0; k <= kmax; k++) {
        if (dtype_abs_squared(dtype_add(Xn,dtype_neg(T[k])))<cycleacc){
            // Compute m = max{i | i < n and ctz(i+1) = k}.
            m = ((((n >> k) - 1) | 1) << k) - 1;
            X[i].real =m-n;
            n=N+1;
        } 
    }
    T[popcount(~((n+1) | -(n+1)))] = Xn; // No match.
    if (dtype_abs_squared(Xn)>DivLim){
        X[i].real=__colouring_mode__;
        n=N+2;
    }   
}
    """

"""Fl seed func, cycles false or 0 for no cycle detection int for number of cycles"""
def PrepStabilityFractalGPU(fl,dtype="cdouble_",cycles=False,ittCountColouring=False,variation=""):
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
    
    Code= Code.replace("__variation_mode__;",variation)
    
    mapclstr = PyToPyOpenCL.subsfunction(flt,Code,"f")
                                                                                                           
    mapclstr=mapclstr.replace("dtype_",dtype)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    if cycles==False or cycles==0:
        return ElementwiseKernel(ctx,dtype+"t"+" *X,int N,double DivLim",mapclstr,"StabilityFractal",preamble="#define PYOPENCL_DEFINE_CDOUBLE //#include <pyopencl-complex.h>  "),queue
    else:
        preamble="""
        #define PYOPENCL_DEFINE_CDOUBLE 
        //#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
        #include <pyopencl-complex.h>
        
        __constant int maxCyclelog2 = """+str(cycles)+"""; // should be set to 32 nearly always
        """
        return ElementwiseKernel(ctx,dtype+"t"+" *X,int N,double DivLim,double cycleacc",mapclstr,"StabilityFractal",preamble=preamble),queue


#@njit   
def OrbitAtPixel(x,y,f,N=8,Divlim=2,variation=None):
    x=complex(x,y)
    Vals=np.zeros(N).astype(np.complex64)
    Vals[0]=f(Vals[1],x)
    for i in range(1,N):
        Vals[i-1]=variation(Vals[i-1])
        Vals[i]=f(Vals[i-1],x)
        if abs(Vals[i])>Divlim:
            return Vals[:i+1]
    return Vals
        
    

def StabilityFractalPyOpenCL(x1,x2,y1,y2,SideLength,mapcl,queue,DivLim=2.0,maxdepth=30,cycles=32,cycleacc=None,shuffle=False):
    DivLim=DivLim**2
    extent = [x1,x2,y1,y2]
    #print(f"{extent} from inside fractgen")
    Stabilities,shape=createstartvals(x1,x2,y1,y2,SideLength)
    Xlength,Ylength=shape
    
    if shuffle:
        orderTemplate = np.arange(SideLength**2)
        shuf_order = orderTemplate
        np.random.shuffle(shuf_order)
        Stabilities=Stabilities[shuf_order]   
    Stabilities=cl.array.to_device(queue,Stabilities)
    if cycles==False or int(cycles) == 0:#not sure about behaviour of not cycles when cycles not a bool
        mapcl(Stabilities,maxdepth,np.float64(DivLim))
    else:
        if cycleacc == None:
            cycleacc=((x1-x2)**2+(y1-y2)**2)/(Xlength**2+Ylength**2)#removed sqrt
            #default threshold  diagonal length of 1 pixel 
            #print(f"cycle tolerance auto calced as {cycleacc}")
        mapcl(Stabilities,maxdepth,np.float64(DivLim),np.float64(cycleacc))
        
    Stabilities=np.real(Stabilities.get())
    if shuffle:
        unshuf_order = np.zeros_like(orderTemplate)
        unshuf_order[shuf_order] = np.arange(SideLength**2)  
        Stabilities=Stabilities[unshuf_order]
    #print(np.min(Stabilities))
    #print(np.max(Stabilities))
    return Stabilities.reshape(Xlength,Ylength),extent

def WrapperOpenCltoDraw(x1,x2,y1,y2,fl,npoints=1000,Divlim=2.0,maxdepth=30,dtype="cdouble_",
                        Code=StabilityFractalNoCycle,cycles=False,cycleacc=1e-5,ittCountColouring=True,
                        variationmode="",ShowOrbits=True):
    fljit=njit(fl,fastmath=True)
    if variationmode == "Burning Ship":
        variationmode=Burningshipvariation
        def Burningship(X):
            return complex(abs(X.real),abs(X.imag))
        def Orbitinnerwrap(x,y):
            return OrbitAtPixel(x,y,fljit,Divlim=Divlim,Divlimvariation=Burningship)
    elif variationmode == "Tricorn":
        variationmode=Tricornpvariation
        def Tricorn(X):
            return np.conj(X)
        def Orbitinnerwrap(x,y):
            return OrbitAtPixel(x,y,fljit,Divlim=Divlim,variation=Tricorn)
    else:

        def identity(X):
            return X
        def Orbitinnerwrap(x,y):
            return OrbitAtPixel(x,y,fljit,Divlim=Divlim,variation=identity)
    openclfunc,queue= PrepStabilityFractalGPU(fl,dtype=dtype,cycles=cycles,ittCountColouring=ittCountColouring,variation=variationmode)
    def innerwrap(x1,x2,y1,y2):
        return StabilityFractalPyOpenCL(x1,x2,y1,y2,npoints,openclfunc,queue,DivLim=Divlim,maxdepth=maxdepth,cycles=cycles,cycleacc=cycleacc)
    
    
    return innerwrap,Orbitinnerwrap

if __name__ == '__main__':#not really intended to be script just here for testing and demo
    x,c=sp.symbols("x,c")
    f=x**2+c
    fl=sp.lambdify((x,c),f)


"""
    mapcl,queueorig=PrepStabilityFractalGPU(fl,cycles=16,ittCountColouring=True)
    queue=queueorig
    k=200
    #starttime = timeit.default_timer()
    max=totaltime=0
    for i in range(k):
        starttime = timeit.default_timer()
        Roots,extent=StabilityFractalPyOpenCL(-2,2,-2,2,1000,mapcl,queue,maxdepth=2000,cycles=16)
        totaltime+=(timeit.default_timer()-starttime)
        if (timeit.default_timer()-starttime)>max:max=(timeit.default_timer()-starttime)
    #SideLength=500
    print(f"total time {totaltime} average {totaltime/k}, longest run {max}")
    
    plt.imshow(Roots,extent=extent)
    plt.show(block=True)"""
"""pre changes:
    total time 20.84446890000001 average 0.10422234450000005, longest run 0.14043470000000013
    Removed sqrt from cycle finder:
    15.368186800000009 average 0.07684093400000004, longest run 0.3338751999999996
    removed some no longer needed bools
    total time 15.128090599999991 average 0.07564045299999995, longest run 0.3307941999999997
    """