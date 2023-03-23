import numpy as np

import sympy as sp
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr
import timeit
from numba import *
from Plotting import ZoomableFractalViewer,StabilityandJuliaDoubleplot
import PyopenclNewtonsFractal
import PyopenclStabilityFractal
np.warnings.filterwarnings("ignore")

x,y,t = sp.symbols('x,y,t')
a,b,c,d,C = sp.symbols('a,b,c,d,C')

"""To do:
fix zoom with newtontype fractals atm some kind of rendering issue or sommin
add second oder newton method fractal
add line fractals such as sepinski triangle"""

def DrawNewtonsFractalOpencl(x1,x2,y1,y2,fl,fprimel=None,npoints=1000, maxdepth=200,tol=1e-16,ShowOrbits=True):
    if isinstance(fl,str):
        fl=parse_expr(fl)
        fp=sp.diff(fl)
        fl=sp.lambdify(x,fl)
        fprimel=sp.lambdify(x,fp)
    if fprimel==None:
        fp=sp.diff(fl)
        fl=sp.lambdify(x,fl)
        fprimel=sp.lambdify(x,fp)
    innerwrap,orbitgen = PyopenclNewtonsFractal.WrapperOpenCltoDraw(x1,x2,y1,y2,fl,fprimel,npoints=npoints, maxdepth=maxdepth,
                                                                    tol=tol,ShowOrbits=ShowOrbits)
    Roots,extent=innerwrap(x1,x2,y1,y2)
    ZoomableFractalViewer(Roots,extent,(innerwrap,orbitgen))
    
def DrawStabilityFractalOpencl(x1,x2,y1,y2,fl,npoints=1024, maxdepth=3000,cycles=16,cycleacc=None,ittcountcolouring=True,Divlim=2,variation="",ShowOrbits=True):
    if isinstance(fl,str):
        fl=parse_expr(fl)
        fl=sp.lambdify((x,c),fl)
    if isinstance(fl,sp.Basic):
        fl=sp.lambdify((x,c),fl)
    
    innerwrap,orbitgen = PyopenclStabilityFractal.WrapperStabilityFractalOpenCltoDraw(x1,x2,y1,y2,fl,npoints=npoints, maxdepth=maxdepth,cycles=cycles,
                                                             cycleacc=cycleacc,ittCountColouring=ittcountcolouring,Divlim=Divlim,
                                                             variationmode=variation,ShowOrbits=ShowOrbits)
    Roots,extent=innerwrap(x1,x2,y1,y2)
    #Roots=np.log(abs(Roots))
    ZoomableFractalViewer(Roots,extent,(innerwrap,orbitgen))
        
def DrawJuliaFractalOpencl(x1,x2,y1,y2,C,fl,npoints=1024, maxdepth=3000,cycles=16,cycleacc=None,ittcountcolouring=True,Divlim=2,variation="",ShowOrbits=True):
    if isinstance(fl,str):
        fl=parse_expr(fl)
        fl=sp.lambdify((x,c),fl)
    if isinstance(fl,sp.Basic):
        fl=sp.lambdify((x,c),fl)
    innerwrap,orbitgen = PyopenclStabilityFractal.WrapperJuliaFractalOpenCltoDraw(x1,x2,y1,y2,C,fl,npoints=npoints, maxdepth=maxdepth,cycles=cycles,
                                                             cycleacc=cycleacc,ittCountColouring=ittcountcolouring,Divlim=Divlim,
                                                             variationmode=variation,ShowOrbits=ShowOrbits) 
    Roots,extent=innerwrap(x1,x2,y1,y2,C)
    ZoomableFractalViewer(Roots,extent,(innerwrap,orbitgen),args=[C])    

def Draw2axisJuliaStabilityFractalOpencl(x1,x2,y1,y2,C,fl,npoints=1024, maxdepth=3000,cycles=16,cycleacc=None,ittcountcolouring=True,Divlim=2,variation="",ShowOrbits=True,x1j=None,x2j=None,y1j=None,y2j=None):
    if isinstance(fl,str):
        fl=parse_expr(fl)
        fl=sp.lambdify((x,c),fl)
    if isinstance(fl,sp.Basic):
        fl=sp.lambdify((x,c),fl)
    if not(all([x1j,x2j,y1j,y2j])):
        x1j,x2j,y1j,y2j=x1,x2,y1,y2   
    #innerwraps contain julliainnerwrap,juliaorbitgen 
    julliainnerwrap = PyopenclStabilityFractal.WrapperJuliaFractalOpenCltoDraw(x1,x2,y1,y2,C,fl,npoints=npoints, maxdepth=maxdepth,cycles=cycles,
                                                             cycleacc=cycleacc,ittCountColouring=ittcountcolouring,Divlim=Divlim,
                                                             variationmode=variation,ShowOrbits=ShowOrbits) 
    JuliaVals,extent2=julliainnerwrap[0](x1j,x2j,y1j,y2j,C)
    #innerwraps contain Stabilityinnerwrap,Stabilityorbitgen
    Stabilityinnerwrap = PyopenclStabilityFractal.WrapperStabilityFractalOpenCltoDraw(x1,x2,y1,y2,fl,npoints=npoints, maxdepth=maxdepth,cycles=cycles,
                                                             cycleacc=cycleacc,ittCountColouring=ittcountcolouring,Divlim=Divlim,
                                                             variationmode=variation,ShowOrbits=ShowOrbits)
    Stabilities,extent=Stabilityinnerwrap[0](x1,x2,y1,y2)
    StabilityandJuliaDoubleplot(Stabilities,JuliaVals,extent,C,Stabilityinnerwrap,julliainnerwrap,extent2=extent2)
    
if __name__ == '__main__':
   
    starttime = timeit.default_timer()
    
    res = 1000
    maxdepth = 1048
    f=x**2+c
    Draw2axisJuliaStabilityFractalOpencl(-2,2,-2,2,-0.4+0.6j,f,maxdepth=maxdepth,npoints=res,cycles=10,ittcountcolouring=True)
    #fl=sp.lambdify(x,f)
    #fpl=sp.lambdify(x,sp.diff(f))
    #DrawStabilityFractalOpencl(-2,2,-2,2,f,maxdepth=maxdepth,npoints=res,cycles=10,ittcountcolouring=True)
    #DrawJuliaFractalOpencl(-2,2,-2,2,-0.4+0.6j,f,maxdepth=maxdepth,npoints=res,cycles=10,ittcountcolouring=True)
    #DrawNewtonsFractalOpencl(-2,2,-2,2,fl,fpl,npoints=res,maxdepth=500,tol=1e-6)
    
    #drawStabilityFractal(npoints=4000,maxdepth=200,ncycles=8)
    #drawnewtontypefractal(f=f,npoints=1000,x1=-1,x2=1,y1=-1,y2=1)
    
    
    #f=x**2-x+1+x**5#example of seed for Newton fractal
    
    