import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
import os
import matplotlib.pyplot as plt
import sympy as sp
import timeit
import PyToPyOpenCL#mine
from Utils import createstartvals#mine
from numba import njit
os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
x,c,y,a,b,d=sp.symbols("x,c,y,a,b,d")
Burningshipvariation="Xn=(dtype_t){fabs(Xn.real),fabs(Xn.imag)};"
Tricornpvariation="Xn=dtype_conj(Xn);"

StabilityFractalNoCycle="""
    int n = 0;
    
    dtype_t Const=X[i];//dont rename
    dtype_t Xn=X[i];
    while (dtype_abs_squared(Xn)<DivLim && n<N) 
    {
        __variation_mode__;
        n+=1;
        Xn=__f(Xn,Const)__;
    }
    __colouring_mode__;
    """
#Gospers algorithm used as cycle detection.
#implementation of gospers algorithm modified from hackers delight. 
StabilityFractalWithCycle="""
short k,kmax,m,n;
dtype_t T[maxCyclelog2+1];


//X[i]= dtype_rmul(0.5,X[i]);
dtype_t Xn=X[i];//breaks julia if set to 0
__JuliaOrMandlebrotlike__
T[0] = X[i];
for (n = 1; n<N; n++) {
    __variation_mode__;
    Xn =__f(Xn,Const,ExtraPrecisionVars[])__;
    ExtraPrecisionVars[0].imag=0;
    ExtraPrecisionVars[0].real=0;
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
def PrepStabilityFractalGPU(fl,dtype="cdouble_",cycles=False,ittCountColouring=False,variation="",juliaMode=False,ExtraPrecisionVars=1):
    
    flt=PyToPyOpenCL.translate(fl,extraPrecisionArgs=ExtraPrecisionVars)
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
    print(flt)
    print(Code)
    mapclstr = PyToPyOpenCL.subsfunction(flt,Code,"f",sig=["x ","c","extrprecision"])                                                                                                      
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    AlwaysArgs=dtype+"t"+" *X,int N,double DivLim"#arguments that will be in every version
    if juliaMode: 
        AlwaysArgs=dtype+"t"+" Const, "+ AlwaysArgs#only required for julia sets
        mapclstr=mapclstr.replace("__JuliaOrMandlebrotlike__","")
        #print(mapclstr)
    else:
        mapclstr=mapclstr.replace("__JuliaOrMandlebrotlike__","dtype_t Const = X[i];")
        print("Julia")
    print(ExtraPrecisionVars)
    if ExtraPrecisionVars>0:
        AlwaysArgs+=", "+dtype+"t"+" *ExtraPrecisionVars"
        ExtraPrecisionVars=np.array([complex(0,0)]*ExtraPrecisionVars)
    else:
        ExtraPrecisionVars=[]
    mapclstr=mapclstr.replace("dtype_",dtype)  
    print("using dtype: ")
    print(mapclstr)  
    if cycles==False or cycles==0:
        return ElementwiseKernel(ctx,AlwaysArgs,mapclstr,"StabilityFractal",preamble="#define PYOPENCL_DEFINE_CDOUBLE //#include <pyopencl-complex.h>  "),queue,ExtraPrecisionVars
    else:
        preamble="""
        #define PYOPENCL_DEFINE_CDOUBLE 
        //#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
        #include <pyopencl-complex.h>
        
        __constant int maxCyclelog2 = """+str(cycles)+"""; // should be set to 32 nearly always
        """
        return ElementwiseKernel(ctx,AlwaysArgs + ",double cycleacc",mapclstr,"StabilityFractal",preamble=preamble),queue,ExtraPrecisionVars



    

def StabilityFractalPyOpenCL(x1,x2,y1,y2,SideLength,mapcl,queue,DivLim=2.0,maxdepth=30,cycles=32,cycleacc=None,shuffle=False,ExtraPrecisionVars=[]):
    DivLim=DivLim**2
    
    extent = np.array([x1,x2,y1,y2])#/Magnification
    print(f"{(x1,x2,y1,y2,SideLength,ExtraPrecisionVars)} from inside fractgen")
    
    
    #(-8.408787292193748e-05, 0.00012886983708417865, -1.4264086909136664, -1.4261653106736594)
    Stabilities,shape,ExtraPrecisionVars,extent=createstartvals(x1,x2,y1,y2,
                                                         SideLength,
                                                         ExtraPrecisionVars=ExtraPrecisionVars)
    print("fractgen after create startvals")
    
    
    #extent=np.array([min(Stabilities.imag),max(Stabilities.imag),min(Stabilities.real),max(Stabilities.real)])
    print(extent)
    x1,x2,y1,y2=extent
    print(f"{(x1,x2,y1,y2,SideLength,ExtraPrecisionVars)} from inside fractgen")
    print(x1+ExtraPrecisionVars[0].real,x2+ExtraPrecisionVars[0].real,y1+ExtraPrecisionVars[0].imag,y2+ExtraPrecisionVars[0].imag)
    print(extent)
    
    for i in range(1,len(Stabilities)):
        if Stabilities[i]==Stabilities[i-1]:
            print(f"XWidth = {x2-x1}")
            print(f"YWidth = {y2-y1}")
            print(extent)
            
            assert False
    Xlength,Ylength=shape
    #if Stabilities[1]-Stabilities[0]<1e-13:
    
    if shuffle:
        orderTemplate = np.arange(SideLength**2)
        shuf_order = orderTemplate
        np.random.shuffle(shuf_order)
        Stabilities=Stabilities[shuf_order]   
    Stabilities=cl.array.to_device(queue,Stabilities)
    ExtraPrecisionVarsin=np.array([complex(ExtraPrecisionVars[0].real,ExtraPrecisionVars[0].imag)])
    ExtraPrecisionVarsin=cl.array.to_device(queue,ExtraPrecisionVarsin)
    if cycles==False or int(cycles) == 0:#not sure about behaviour of not cycles when cycles not a bool
        mapcl(Stabilities,maxdepth,np.float64(DivLim))
    else:
        if cycleacc == None:
            cycleacc=((x1-x2)**2+(y1-y2)**2)/(Xlength**2+Ylength**2)#removed sqrt
            #default threshold  diagonal length of 1 pixel 
            #print(f"cycle tolerance auto calced as {cycleacc}")
        
        #ExtraPrecisionVarsin=ExtraPrecisionVars.conjugate()
        print(ExtraPrecisionVarsin)
        mapcl(Stabilities,maxdepth,np.float64(DivLim),ExtraPrecisionVarsin,np.float64(cycleacc))
        
    Stabilities=np.real(Stabilities.get())
    #ExtraPrecisionVars=ExtraPrecisionVars.get()
    if shuffle:
        unshuf_order = np.zeros_like(orderTemplate)
        unshuf_order[shuf_order] = np.arange(SideLength**2)  
        Stabilities=Stabilities[unshuf_order]
    #print(np.min(Stabilities))
    #print(np.max(Stabilities))
    if abs(ExtraPrecisionVars[0])!=0:
                plt.imshow(Stabilities.reshape(Xlength,Ylength))
                plt.show(block=True)
    print(type(extent[0]))
    print(type(extent))
    print("just before leaving fractgen")
    print(extent)
    return Stabilities.reshape(Xlength,Ylength),(extent,ExtraPrecisionVars)

def JuliaFractalPyOpenCL(x1,x2,y1,y2,C,SideLength,mapcl,queue,DivLim=2.0,maxdepth=30,cycles=32,cycleacc=None):
    DivLim=DivLim**2
    extent = [x1,x2,y1,y2]
    #print(f"{extent} from inside fractgen")
    Stabilities,shape=createstartvals(x1,x2,y1,y2,SideLength)
    Xlength,Ylength=shape
    
    Stabilities=cl.array.to_device(queue,Stabilities)
    if cycles==False or int(cycles) == 0:#not sure about behaviour of not cycles when cycles not a bool
        
        mapcl(C,Stabilities,maxdepth,np.float64(DivLim))
    else:
        if cycleacc == None:
            cycleacc=((x1-x2)**2+(y1-y2)**2)/(Xlength**2+Ylength**2)#removed sqrt
            #default threshold  diagonal length of 1 pixel 
            #print(f"cycle tolerance auto calced as {cycleacc}")
        print((C,Stabilities,maxdepth,np.float64(DivLim),np.float64(cycleacc)))
        
        mapcl(C,Stabilities,maxdepth,np.float64(DivLim),np.float64(cycleacc))
       
    Stabilities=np.real(Stabilities.get())
    #print(np.min(Stabilities))
    #print(np.max(Stabilities))
    return Stabilities.reshape(Xlength,Ylength),extent

#@njit   
def OrbitAtPixel(x,y,f,N=8,Divlim=2,variation=None):
    x=complex(x,y)
    
    Vals=np.zeros(N).astype(np.complex64)
    Vals[0]=x
    Vals[1]=f(Vals[0],x)
    for i in range(2,N):
        Vals[i-1]=variation(Vals[i-1])
        Vals[i]=f(Vals[i-1],x)
        if abs(Vals[i])>Divlim:
            return Vals[:i+1]
    return Vals
        
def JuliaOrbitAtPixel(x,y,c,f,N=8,Divlim=2,variation=None):
    x=complex(x,y)
    
    Vals=np.zeros(N).astype(np.complex64)
    Vals[0]=x
    Vals[1]=f(Vals[0],c)
    for i in range(2,N):
        Vals[i-1]=variation(Vals[i-1])
        Vals[i]=f(Vals[i-1],c)
        if abs(Vals[i])>Divlim:
            return Vals[:i+1]
    return Vals

def applyvariation(variationmode,fl,Divlim):
    
    fljit=njit(fl,fastmath=True)
    if variationmode == "Burning Ship":
        variationmode=Burningshipvariation
        def Burningship(X):
            return complex(abs(X.real),abs(X.imag))
        def Orbitinnerwrap(x,y):
            return OrbitAtPixel(x,y,fljit,Divlim=Divlim,variation=Burningship)
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
    return Orbitinnerwrap,variationmode

def Juliaapplyvariation(variationmode,fl,Divlim):
    print(variationmode)
    fljit=njit(fl,fastmath=True)
    if variationmode == "Burning Ship":
        variationmode=Burningshipvariation
        def Burningship(X):
            return complex(abs(X.real),abs(X.imag))
        def Orbitinnerwrap(x,y,c):
            return JuliaOrbitAtPixel(x,y,c,fljit,Divlim=Divlim,variation=Burningship)
    elif variationmode == "Tricorn":
        variationmode=Tricornpvariation
        def Tricorn(X):
            return np.conj(X)
        def Orbitinnerwrap(x,y,c):
            return JuliaOrbitAtPixel(x,y,c,fljit,Divlim=Divlim,variation=Tricorn)
    else:
        def identity(X):
            return X
        def Orbitinnerwrap(x,y,c):
            return JuliaOrbitAtPixel(x,y,c,fljit,Divlim=Divlim,variation=identity)
    return Orbitinnerwrap,variationmode

def WrapperStabilityFractalOpenCltoDraw(x1,x2,y1,y2,fl,npoints=1000,Divlim=2.0,maxdepth=30,dtype="cdouble_",
                        Code=StabilityFractalNoCycle,cycles=False,cycleacc=1e-5,ittCountColouring=True,
                        variationmode="",ShowOrbits=True,ExtraPrecisionVars=0):
    
    
    openclfunc,queue,ExtraPrecisionVars= PrepStabilityFractalGPU(fl,
                                                                 dtype=dtype,
                                                                 cycles=cycles,
                                                                 ittCountColouring=ittCountColouring,
                                                                 variation=variationmode,
                                                                 ExtraPrecisionVars=ExtraPrecisionVars)
    
    if isinstance(fl,sp.Basic):
        fl=sp.lambdify((x,c),fl)
    Orbitinnerwrap,variationmode=applyvariation(variationmode,fl,Divlim)
    def innerwrap(x1,x2,y1,y2,maxdepth=maxdepth,ExtraPrecisionVars=ExtraPrecisionVars):
        return StabilityFractalPyOpenCL(x1,x2,y1,y2,npoints,openclfunc,queue,ExtraPrecisionVars=ExtraPrecisionVars,DivLim=Divlim,maxdepth=maxdepth,cycles=cycles,cycleacc=cycleacc)
    return innerwrap,Orbitinnerwrap,ExtraPrecisionVars

def WrapperJuliaFractalOpenCltoDraw(x1,x2,y1,y2,C,fl,npoints=1000,Divlim=2.0,maxdepth=30,dtype="cdouble_",
                        Code=StabilityFractalNoCycle,cycles=False,cycleacc=1e-5,ittCountColouring=True,
                        variationmode="",ShowOrbits=True):
    Orbitinnerwrap,variationmode=Juliaapplyvariation(variationmode,fl,Divlim)
    openclfunc,queue,ExtraPrecisionVars= PrepStabilityFractalGPU(fl,dtype=dtype,cycles=cycles,ittCountColouring=ittCountColouring,variation=variationmode,juliaMode=True)
    def innerwrap(x1,x2,y1,y2,C,maxdepth=maxdepth):
        return JuliaFractalPyOpenCL(x1,x2,y1,y2,C,npoints,openclfunc,queue,DivLim=Divlim,maxdepth=maxdepth,cycles=cycles,cycleacc=cycleacc)
    return innerwrap,Orbitinnerwrap

if __name__ == '__main__':#not really intended to be script just here for testing and demo
    x,c=sp.symbols("x,c")
    f=x**2+c
    fl=sp.lambdify((x,c),f)
    mapcl,queueorig=PrepStabilityFractalGPU(fl,cycles=0,ittCountColouring=True)
    queue=queueorig
    D=2
    #Roots1,extent=StabilityFractalPyOpenCL(-2,2,-2,2,1000,mapcl,queue,maxdepth=D,cycles=16)
    #fig,axs=plt.subplots(3)
    extent=np.array([-2,2,-2,2])
    """Roots1pre,notextent=StabilityFractalPyOpenCL(*(np.array(extent)),500,mapcl,queue,maxdepth=2000,cycles=0)
    axs[0].imshow(Roots1pre)
    Roots2pre,notextent=StabilityFractalPyOpenCL(*(np.array(extent)),500,mapcl,queue,maxdepth=2002,cycles=0)
    axs[1].imshow(Roots2pre)
    print(np.sum(np.abs(Roots1pre-Roots2pre)))
    axs[2].imshow(np.abs(Roots1pre-Roots2pre))
    plt.show()"""
    diffs=[]
    depths=[]
    diffs2=[]
    diffs3=[]
    diffs4=[]
    
    
    Roots1pre,notextent=StabilityFractalPyOpenCL(*(np.array(extent)),800,mapcl,queue,maxdepth=D,cycles=0)
        
    Roots2pre,notextent=StabilityFractalPyOpenCL(*(np.array(extent)*(1/2)),800,mapcl,queue,maxdepth=D,cycles=0)
        
    Roots3pre,notextent=StabilityFractalPyOpenCL(*(np.array(extent)*(1/4)),800,mapcl,queue,maxdepth=D,cycles=0)
        
    #Roots4pre,notextent=StabilityFractalPyOpenCL(*(np.array(extent)*(1/8)),800,mapcl,queue,maxdepth=D,cycles=0)
    for i in range(1,40):
        Roots1,notextent=StabilityFractalPyOpenCL(*(np.array(extent)),800,mapcl,queue,maxdepth=D+1*i,cycles=0)
        
        Roots2,notextent=StabilityFractalPyOpenCL(*(np.array(extent)*(1/2)),800,mapcl,queue,maxdepth=D+1*i,cycles=0)
        
        Roots3,notextent=StabilityFractalPyOpenCL(*(np.array(extent)*(1/4)),800,mapcl,queue,maxdepth=D+1*i,cycles=0)
        
        #Roots4,notextent=StabilityFractalPyOpenCL(*(np.array(extent)*(1/8)),800,mapcl,queue,maxdepth=D+1*i,cycles=0)
        
        diffs.append(np.sum(Roots1-Roots1pre))
        depths.append(D+1*i)
        diffs2.append(np.sum(Roots2-Roots2pre))
        diffs3.append(np.sum(Roots3-Roots3pre))
        #diffs4.append(np.sum(Roots4-Roots4pre))
        Roots1pre=Roots1
        Roots2pre=Roots2
        Roots3pre=Roots3
        if len(diffs)>=4:
            if diffs[i-2]-diffs[i-3]==diffs[i-1]-diffs[i-2]:
                print("1")
                print(i)
            if diffs2[i-2]-diffs2[i-3]==diffs2[i-1]-diffs2[i-2]:
                print("2")
                print(i)
            if diffs3[i-2]-diffs3[i-3]==diffs3[i-1]-diffs3[i-2]:
                print("3")
                print(i)
                
        #Roots4pre=Roots4
    fig,axs=plt.subplots(3)
    axs[0].plot(depths,np.abs(diffs))
    axs[1].plot(np.log(depths),np.log(np.abs(diffs)))
    axs[2].plot(depths,diffs)
    axs[0].plot(depths,np.abs(diffs2))
    axs[1].plot(np.log(depths),np.log(np.abs(diffs2)))
    axs[2].plot(depths,diffs2)
    axs[0].plot(depths,np.abs(diffs3))
    axs[1].plot(np.log(depths),np.log(np.abs(diffs3)))
    axs[2].plot(depths,diffs3)
    
    print(diffs)
  
    plt.show()
    """max=totaltime=0
    for i in range(k):
        starttime = timeit.default_timer()
        Roots,extent=StabilityFractalPyOpenCL(-2,2,-2,2,1000,mapcl,queue,maxdepth=2000,cycles=16)
        totaltime+=(timeit.default_timer()-starttime)
        if (timeit.default_timer()-starttime)>max:max=(timeit.default_timer()-starttime)"""

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