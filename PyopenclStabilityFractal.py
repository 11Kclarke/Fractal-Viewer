import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
import os
import matplotlib.pyplot as plt
import sympy as sp
import timeit
import PyToPyOpenCL#mine
from Utils import CreateStartVals#mine
from numba import njit
os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

Burningshipvariation="Xn=(dtype_t){fabs(Xn.real),fabs(Xn.imag)};"
Tricornpvariation="Xn=dtype_conj(Xn);"


#Gospers algorithm used as cycle detection.
#implementation of gospers algorithm modified from hackers delight. 
StabilityFractalWithCycle="""
short k,kmax,m,n;
dtype_t T[maxCyclelog2+1];
//dtype_t Xn=X[i];//should prolly be 0
//X[i]= dtype_rmul(0.5,X[i]);
__JuliaOrMandlebrotlike__
//dtype_t Xn=(dtype_t){0.0,0.0};
dtype_t Xn=X[i];
T[0] = X[i];
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
StabilityFractalPerturbation="""
short k,kmax,m,n;
dtype_t T[maxCyclelog2+1];

//dtype_t diff0 = dtype_add(X[i],dtype_neg(ReferenceOrbit[0]));//X0-referenceorbit0 = delta_0
dtype_t diff0=X[i];
//dtype_t Constpert=dtype_add(diff0,ReferenceOrbit[0]);//Y0 start of actual orbit
dtype_t diffn=diff0;
bool PerturbationModeFailed = false;
//dtype_t A = dtype_addr(dtype_rmul(2,ReferenceOrbit[0]),1);//ABC_1
//dtype_t B =(dtype_t){1.0,0.0};
//dtype_t C =(dtype_t){0.0,0.0};
int refitt =1;
int RebaseCount =0;
__JuliaOrMandlebrotlike__
dtype_t Xn=X[i];//breaks julia if 0
T[0] = (dtype_t){0.0,0.0};
for (n = 1; n<N-1; n++) {
    
    __variation_mode__;
    if ((dtype_abs_squared(ReferenceOrbit[refitt+1])==0))// && dtype_abs(diffn)<1e-10)dtype_abs_squared(Xn)<dtype_abs_squared(diffn) || 
    {
        diffn=Xn;
        refitt =1;
        RebaseCount+=1;
        if (RebaseCount >100)
        {
            n=N+2;
            
            printf("Exceeded max rebase");
        }
        //PerturbationModeFailed=true;
        //Xn =__f(Xn,Constpert)__;
    }
    else
    {
    
    diffn=dtype_add(dtype_add(dtype_mul(dtype_rmul(2,ReferenceOrbit[refitt]),diffn),dtype_powr(diffn,2)),diff0);
    Xn=dtype_add(diffn,ReferenceOrbit[refitt]);   
    }
    refitt++;
    
        
    //Xn=dtype_add(diffn,ReferenceOrbit[n]);
    
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
    
    //A=dtype_addr(dtype_mul(dtype_rmul(2,ReferenceOrbit[n]),A),1);
    //A^2+2*Reference_n*B
    
    
    //B=dtype_add(dtype_mul(dtype_rmul(2,ReferenceOrbit[n]),B),dtype_mul(A,A));
    //B*A*2+2*Reference_n*C
    
    //C=dtype_add(dtype_mul(dtype_rmul(2,ReferenceOrbit[n]),C),dtype_mul(dtype_rmul(2,A),B));
    //C*diff^3+B*diff^2+A*diff
   
    //diffn=dtype_add(dtype_add(dtype_mul(A,diff0),dtype_mul(B,dtype_powr(diff0,2))),dtype_mul(C,dtype_powr(diff0,3)));   
}
    """

def ApplySettingToKernel(Code,ittCountColouring,Variation,JuliaMode,dtype):
    if ittCountColouring:   
        Code= Code.replace("__colouring_mode__;","X[i].real = n;")
    else:
        Code= Code.replace("__colouring_mode__;","X[i].real = Xn.real;")#Comments the code for colouring by ittcount   
    Code= Code.replace("__variation_mode__;",Variation)
    if JuliaMode: 
        AlwaysArgs=dtype+"t"+" Const, "+ AlwaysArgs#only required for julia sets
        Code=Code.replace("__JuliaOrMandlebrotlike__","")
    else:
        Code=Code.replace("__JuliaOrMandlebrotlike__","dtype_t Const = X[i];")
    Code=Code.replace("dtype_",dtype)
    return Code
"""Fl seed func, cycles false or 0 for no cycle detection int for number of cycles"""
def PrepStabilityFractalGPU(fl,dtype="cdouble_",cycles=False,ittCountColouring=False,variation="",JuliaMode=False,DivLim=2.0,maxdepth=30):
    if isinstance(fl,str):
        fl=sp.lambdify(x,fl)
    
    Orbitinnerwrap,variation=applyvariation(variation,fl,DivLim)
    
    def startvalswithfunc(x1,x2,y1,y2,SideLength,ReferenceOrbit=None):
        return CreateStartVals(x1,x2,y1,y2,SideLength,
                               Func=fl,
                               DivLim=DivLim,
                               maxdepth=maxdepth,
                               ReferenceOrbit=ReferenceOrbit)
        
    flt=PyToPyOpenCL.translate(fl)
    cl.tools.get_or_register_dtype(dtype+"t", np.complex128)
    Standard=ApplySettingToKernel(StabilityFractalWithCycle,ittCountColouring,variation,JuliaMode,dtype)
    Perturbation=ApplySettingToKernel(StabilityFractalPerturbation,ittCountColouring,variation,JuliaMode,dtype)
    Standard = PyToPyOpenCL.subsfunction(flt,Standard,"f")  
    Perturbation = PyToPyOpenCL.subsfunction(flt,Perturbation,"f")                                                                                                    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    AlwaysArgs=dtype+"t"+" *X,int N,double DivLim,double cycleacc"#arguments that will be in every version
    preamble="""
    #define PYOPENCL_DEFINE_CDOUBLE 
    //#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    #include <pyopencl-complex.h>
        
    __constant int maxCyclelog2 = """+str(cycles)+"""; // should be set to 32 nearly always
        """
    StandardKernel=ElementwiseKernel(ctx,AlwaysArgs ,Standard,"StabilityFractal",preamble=preamble)
    PerturbationKernel=ElementwiseKernel(ctx,AlwaysArgs + ","+dtype+"t"+ "*ReferenceOrbit",Perturbation,"StabilityFractal",preamble=preamble)
    return (StandardKernel,PerturbationKernel),queue,startvalswithfunc,Orbitinnerwrap



    

def StabilityFractalPyOpenCL(x1,x2,y1,y2,SideLength,mapcl,queue,DivLim=2.0,maxdepth=30,cycles=32,cycleacc=None,CreateStartvals=CreateStartVals,ReferenceOrbit=None):
    DivLim=DivLim**2
    
    extent = [x1,x2,y1,y2]
    print(f"extent {extent} from inside fractgen pre Creatstartvals")
    InnitialValues,shape,ReferenceOrbit=CreateStartvals(x1,x2,y1,y2,SideLength,ReferenceOrbit=ReferenceOrbit)
    Xlength,Ylength=shape
    if cycleacc == None:
        cycleacc=((x1-x2)**2+(y1-y2)**2)/(Xlength**2+Ylength**2)#removed sqrt
            #default threshold  diagonal length of 1 pixel 
            #print(f"cycle tolerance auto calced as {cycleacc}")
    x1=np.min(np.real(InnitialValues))
    x2=np.max(np.real(InnitialValues))
    y1=np.min(np.imag(InnitialValues))
    y2=np.max(np.imag(InnitialValues))
    extent = [x1,x2,y1,y2]
    print(f"extent {extent} from inside fractgen post Creatstartvals")
    mapcl,mapclPerturbation=mapcl    
    
    Stabilities=InnitialValues.copy()
    InnitialValues=cl.array.to_device(queue,InnitialValues)
    
    
    
    
    if isinstance(ReferenceOrbit,np.ndarray):
        #print(ReferenceOrbit)
        ReferenceOrbitcopy=ReferenceOrbit.copy()
        ReferenceOrbit=cl.array.to_device(queue,ReferenceOrbit)
        mapclPerturbation(InnitialValues,maxdepth,np.float64(DivLim),np.float64(cycleacc),ReferenceOrbit)
        ReferenceOrbit.get(ary=ReferenceOrbitcopy)
        ReferenceOrbit=ReferenceOrbitcopy
        #print(ReferenceOrbit)
    else:
        mapcl(InnitialValues,maxdepth,np.float64(DivLim),np.float64(cycleacc))
       
    InnitialValues.get(ary=Stabilities)
    Stabilities=Stabilities.real.reshape(Xlength,Ylength)
    
    print(f"extent {extent} from inside fractgen just before return")
    return Stabilities,(extent,ReferenceOrbit)

def JuliaFractalPyOpenCL(x1,x2,y1,y2,C,SideLength,mapcl,queue,DivLim=2.0,maxdepth=30,cycles=32,cycleacc=None):
    DivLim=DivLim**2
    extent = [x1,x2,y1,y2]
    #print(f"{extent} from inside fractgen")
    InnitialVals,shape=CreateStartVals(x1,x2,y1,y2,SideLength)
    Xlength,Ylength=shape
    Stabilities=InnitialVals.copy()
    InnitialVals=cl.array.to_device(queue,InnitialVals)
    if cycles==False or int(cycles) == 0:#not sure about behaviour of not cycles when cycles not a bool
        mapcl(C,InnitialVals,maxdepth,np.float64(DivLim))
    else:
        if cycleacc == None:
            cycleacc=((x1-x2)**2+(y1-y2)**2)/(Xlength**2+Ylength**2)#removed sqrt
            #default threshold  diagonal length of 1 pixel 
            #print(f"cycle tolerance auto calced as {cycleacc}")
        print((C,InnitialVals,maxdepth,np.float64(DivLim),np.float64(cycleacc)))
        mapcl(C,InnitialVals,maxdepth,np.float64(DivLim),np.float64(cycleacc))
        
    InnitialVals.get(ary=Stabilities)
    #print(np.min(Stabilities))
    #print(np.max(Stabilities))
    return Stabilities.real.reshape(Xlength,Ylength),extent

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

def WrapperStabilityFractalOpenCltoDraw(x1,x2,y1,y2,fl,npoints=1000,DivLim=2.0,maxdepth=30,dtype="cdouble_",
                        Code=StabilityFractalWithCycle,cycles=False,cycleacc=1e-5,ittCountColouring=True,
                        variationmode="",ShowOrbits=True,ReferenceOrbit=None):
    #Orbitinnerwrap,variationmode=applyvariation(variationmode,fl,DivLim)
    openclfunc,queue,startvalswithfunc,Orbitinnerwrap= PrepStabilityFractalGPU(fl,
                                                                dtype=dtype,
                                                                cycles=cycles,
                                                                ittCountColouring=ittCountColouring,
                                                                variation=variationmode,
                                                                DivLim=DivLim,
                                                                maxdepth=maxdepth)
    def innerwrap(x1,x2,y1,y2,maxdepth=maxdepth,ReferenceOrbit=ReferenceOrbit):
        return StabilityFractalPyOpenCL(x1,x2,y1,y2,
                                        npoints,
                                        openclfunc,
                                        queue,
                                        DivLim=DivLim,
                                        maxdepth=maxdepth,
                                        cycles=cycles,
                                        cycleacc=cycleacc,
                                        CreateStartvals=startvalswithfunc,
                                        ReferenceOrbit=ReferenceOrbit)
    return innerwrap,Orbitinnerwrap

def WrapperJuliaFractalOpenCltoDraw(x1,x2,y1,y2,C,fl,npoints=1000,DivLim=2.0,maxdepth=30,dtype="cdouble_",
                        Code=StabilityFractalWithCycle,cycles=False,cycleacc=1e-5,ittCountColouring=True,
                        variationmode="",ShowOrbits=True):
    Orbitinnerwrap,variationmode=Juliaapplyvariation(variationmode,fl,DivLim)
    openclfunc,queue,startvalswithfunc= PrepStabilityFractalGPU(fl,
                                                                dtype=dtype,
                                                                cycles=cycles,
                                                                ittCountColouring=ittCountColouring,
                                                                variation=variationmode,
                                                                juliaMode=True,
                                                                createstartvals=startvalswithfunc,
                                                                DivLim=DivLim,
                                                                maxdepth=maxdepth)
    def innerwrap(x1,x2,y1,y2,C,maxdepth=maxdepth):
        return JuliaFractalPyOpenCL(x1,x2,y1,y2,C,npoints,openclfunc,queue,DivLim=DivLim,maxdepth=maxdepth,cycles=cycles,cycleacc=cycleacc)
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