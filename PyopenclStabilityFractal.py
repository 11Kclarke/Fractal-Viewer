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
StabilityFractalNoCycle="""
    int Counter = 0;
    dtype_t Const=X[i];//dont rename
    bool notfound = true;
    while (dtype_abs(X[i])<DivLim && Counter<N) 
    {
        Counter+=1;
        X[i]=__f(X[i],Const)__;
    }
    """
    
StabilityFractalWithCycle="""
    int Counter = 0;
    dtype_t Const=X[i];//dont rename
    bool notfound = true;
    dtype_t pastvals [maxcycle];
    while (dtype_abs_squared(X[i])<DivLim && Counter<N){
        Counter+=1;
        X[i]=__f(X[i],Const)__;
        pastvals[Counter%maxcycle]=X[i];
        //Counter%maxcycle start for cycle finding
        for (int j = 1; j < maxcycle; ++j)//goes backwards through stored vals 
        {
                if (dtype_abs_squared(dtype_add(X[i],dtype_neg(pastvals[(Counter-j)%maxcycle])))<cycleacc)
                {
                X[i].real=-j-1;//negative so it can be seen easily where cycles have been found
                Counter=N+1;
                PYOPENCL_ELWISE_CONTINUE;//pretty sure this does nothing
                }
            }
    }
    
    """
os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

"""Fl seed func, cycles false or 0 for no cycle detection int for number of cycles"""
def PrepStabilityFractalGPU(fl,dtype="cdouble_",cycles=False,ittCountColouring=False):
    if isinstance(fl,str):
        fl=sp.lambdify(x,fl)
    flt=PyToPyOpenCL.translate(fl)
    cl.tools.get_or_register_dtype(dtype+"t", np.complex128)
    
    CodeForIttCountColouring="""
    if(Counter<N)
    {
        X[i].real=Counter;  
    }
    else if(Counter==N)
    {
        X[i].real=0;
    }
        
       
        """
    if cycles==False or cycles==0:#not sure about behaviour of not cycles when cycles not a bool
        Code=StabilityFractalNoCycle    
    elif  isinstance(cycles,int):
        Code=StabilityFractalWithCycle
        
    if ittCountColouring:
        Code+=CodeForIttCountColouring#adds the code for colouring by ittcount  
    if isinstance(cycles,str):
        Code=cycles
    
    
    mapclstr = PyToPyOpenCL.subsfunction(flt,Code,"f")
    print(mapclstr)                                                                                                          
    mapclstr=mapclstr.replace("dtype_",dtype)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    if Code==StabilityFractalNoCycle:return ElementwiseKernel(ctx,dtype+"t"+" *X,int N,double DivLim",mapclstr,"StabilityFractal",preamble="#define PYOPENCL_DEFINE_CDOUBLE //#include <pyopencl-complex.h>  "),queue
    else:
        preamble="""
        #define PYOPENCL_DEFINE_CDOUBLE 
        #include <pyopencl-complex.h>
        __constant int maxcycle = """+str(cycles)+""";
        """
        return ElementwiseKernel(ctx,dtype+"t"+" *X,int N,double DivLim,double cycleacc",mapclstr,"StabilityFractal",preamble=preamble),queue
    

def StabilityFractalPyOpenCL(x1,x2,y1,y2,SideLength,mapcl,queue,Divlim=2.0,maxdepth=30,cycles=False,cycleacc=None,shuffle=False):
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
        print(shuffletime)
    Stabilities=cl.array.to_device(queue,Stabilities)
    
    Xlength=int(np.ceil(abs((x1-x2)/(y1-y2)*SideLength)))
    Ylength=int(np.ceil((SideLength**2)/Xlength))
    
    if cycles==False or cycles == 0:#not sure about behaviour of not cycles when cycles not a bool
        
        mapcl(Stabilities,maxdepth,np.float64(Divlim))
    else:
        if cycleacc == None:
            cycleacc=(((x1-x2)**2+(y1-y2)**2)/(Xlength**2+Ylength**2))
            #default threshold is 50% of diagonal length of 1 pixel 
            #print(f"cycle tolerance auto calced as {cycleacc}")
        Time=timeit.default_timer()
        mapcl(Stabilities,maxdepth,np.float64(Divlim),np.float64(cycleacc))
        
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
    #print(np.min(Stabilities))
    #print(np.max(Stabilities))
    #print(maxdepth)
    print(f"whole generatorfucntion time {timeit.default_timer()-origTime}")
    return Stabilities.reshape(Xlength,Ylength),extent

def WrapperOpenCltoDraw(x1,x2,y1,y2,fl,npoints=1000,Divlim=2.0,maxdepth=30,dtype="cdouble_"
                    ,Code=StabilityFractalNoCycle,cycles=False,cycleacc=1e-5,ittCountColouring=True):
    openclfunc,queue= PrepStabilityFractalGPU(fl,dtype=dtype,cycles=cycles,ittCountColouring=ittCountColouring)
    def innerwrap(x1,x2,y1,y2):
        return StabilityFractalPyOpenCL(x1,x2,y1,y2,npoints,openclfunc,queue,Divlim=Divlim,maxdepth=maxdepth,cycles=cycles,cycleacc=cycleacc)
    return innerwrap

if __name__ == '__main__':#not really intended to be script just here for testing and demo
    x,Const=sp.symbols("x,Const")
    f=x**2+Const
    fl=sp.lambdify((x,Const),f)


   
    
    SideLength=500
    
    mapcl,queue=PrepStabilityFractalGPU(fl,cycles=False,ittCountColouring=False)
    starttime = timeit.default_timer()
    Roots,extent=StabilityFractalPyOpenCL(-2,2,2,-2,1000,mapcl,queue,cycles=False)
    print(timeit.default_timer()-starttime)
    
    plt.imshow(Roots,extent=extent)
    plt.show(block=True)