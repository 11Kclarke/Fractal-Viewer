import sys
print(sys.version)
from tokenize import Exponent
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr
import sys
import timeit
import math
from matplotlib.widgets import TextBox
import numba
from numba import *


@njit("complex128(complex128,complex128,complex128,complex128,complex128)",fastmath=True)
def generalcomplexpoly(Zn,d,a=2,b=2,power=2):
    #returns value of complex polynomial of the form z^power+a*Zn+b*j+d
    return Zn**power+a*Zn+b*1j+d

@njit("complex128(complex128,complex128)",fastmath=True)
def mandlebrot(Zn,C):
    pow=2
    return Zn**pow+C

#cool, move to right 
@njit("complex128(complex128,complex128)",fastmath=True)
def exponential(Zn,C):
    
    return np.exp(Zn+C)

#some mandlebrot style seed functions
#checkout seed funcs.py for more



@njit("boolean(complex128,int16)")
def absdivergencedetectoroptimised(ys,val):
 if abs(ys)>val:
     return True
 return False


@njit("Tuple((boolean,int16))(complex128[:],float64,int32,int32)",fastmath=True,parallel=True,nogil = True)
def cycledetector(ys,D,cycles,j):
    #if abs(ys[2]-ys[0])<1e-4:
    #if math.isclose(abs(ys[2]),abs(ys[0]),rel_tol=1e-6):
    D=D*(1-1e-2)
    for i in prange(1,cycles):
        if i == j:
            continue
        if abs(ys[i]-ys[j])<D:
            return True,-i
    return (False,0)


@njit(fastmath = True)
def newtonsmethod(f,fprime,x0,tol=1e-16,N=4,maxdepth=500):#,f=None,fprime=None):
    #print(f)
    #print(fprime)
    #print(N)
    #print(tol)
    x=x0+0j
    for i in range(maxdepth):
        x=x-f(x)/fprime(x)
        if abs(f(x))<tol:
            root=np.float64(round(abs(x.real+x.imag),-(N+2)))#convert to real number that is different from its complex conjugate
            #root = np.float64(int(abs(x.real+x.imag)*10**-(N+1))*10**-(N+1))
            #preferably as different as possible from the other roots
            #this is required as matplotlib canot colourise complex numbers
            return root
    return 0 

   
 

@njit(fastmath = True,parallel=True,nogil = True)
def newtonsfractalfornjit(x1,x2,y1,y2,npoints=600, maxdepth=80,tol=1e-16,iterator=newtonsmethod,Nroots=None):#,f=None,dydx=None):
    #print(type(tol))
    #print(str(tol))
    
    #tol=(((x1-x2)/npoints)**2+((y1-y2)/npoints)**2)/4
    N=int(np.log10(tol))
    flatroots = np.ones((npoints,npoints),dtype=np.float64).flatten()#cannot be dtype complex as it will break the colouringa
    extent = [x1,x2,y1,y2]
    xvals = np.linspace(x1*1j,x2*1j,npoints)
    yvals = np.linspace(y1+0j,y2+0j,npoints)
    
    #put in 1 loop as it parralelizes easier
    for c in prange(0,npoints*npoints):
            flatroots[c]=iterator(xvals[c//npoints]+yvals[c%npoints],tol=tol,N=N)
            #flatroots[c]=iterator(xvals[c//npoints]+yvals[c%npoints])       
    #for this algorithm to work difference between roots needs to be greater then difference between different copies of same root
    #this wont be fulfilled in the case of stacked roots for example X^5 has 5 roots exactly at 0 and will find 5 different rounding errors
    roots=flatroots.reshape(npoints,npoints)
    
    """if Nroots != None:
        np.sort(roots.flatten()).reshape(npoints,npoints)
        indexofroots = np.argpartition(np.diff(roots), -Nroots+1)[-Nroots+1:]#gets index of largest diffs between adjacent values
        uniqueroots=np.zeros(Nroots)
        uniqueroots[:-1]=np.array(roots[indexofroots])
        uniqueroots[-1]=roots[-1]#values for roots all chosen
        #indexofroots.sort()
        roots[0:indexofroots[0]]=roots[indexofroots[0]]
        for i in range(1,Nroots-1):
            roots[indexofroots[i-1]:indexofroots[i]]=roots[indexofroots[i]]
            print("known number of roots")
        roots=roots[roots.argsort()]    
        print("known number of roots")
        return roots"""
    print("\n\n Roots:\n")
    print(roots)

    print(len(np.unique(roots)))
    print(np.unique(roots))
    print("\n\n")
    return roots,extent



"""function should be callable, in form of lamdified sympy
divergence detector should return True if diverged"""
@njit(fastmath = True)
def iterator(function, maxdepth, value,D,divergencedetector=absdivergencedetectoroptimised,cycledetector=cycledetector,divlim=2,cycles=32):
    ys=np.zeros(cycles,dtype=np.complex128)
    ys[0]=value
    for i in range(1,cycles):#cycles interger
        ys[i]=function(ys[i-1],value)#stores cycles number of previous values, in order to find cycles
    if divergencedetector(ys[-1],divlim):
        return cycles
   
    for i in range(cycles,maxdepth):
        ys[i%cycles]=function(ys[i%cycles-1],value)#forms queue of past values 
        cycle,period=cycledetector(ys,D,cycles,i%cycles)#checks for cycle
        if cycle:
            return period
        if divergencedetector(ys[i%cycles],divlim):
            return i
    return maxdepth+10

"""Takes complex to complex function, and complex value, 
and returns reecursion count before value > lim"""
@njit(fastmath = True)
def simprecurse(function, maxdepth, value,D,divergencedetector=absdivergencedetectoroptimised,cycledetector=cycledetector,divlim=2,cycles=None):
    #divergencedetector(0,1)
    
    ys=function(0,value)
    for i in range(maxdepth-1):
        if abs(ys)>divlim: 
            return i                        
        ys=function(ys,value)
    return maxdepth+10


   






"""Does the same as genfractfromvertex, but is compatible with @njit"""
@njit(fastmath = True,parallel=True,nogil = True)
def genfractfromvertexfornjit(f,x1,x2,y1,y2,npoints=600, maxdepth=1000,iscomplex=True,iterator=simprecurse):
    #print("Generating fractal from vertex")
    flatstabilities = np.ones((npoints*npoints),dtype=np.float64)*maxdepth#makes flat array for storing stabilities
    dx=(x1-x2)/npoints
    dy=(y1-y2)/npoints
    extent = [x1+dx,x2-dx,y1+dy,y2-dy]
    D=math.sqrt(dx**2+dy**2)
    xvals = np.linspace(x1*1j,x2*1j,npoints)
    yvals = np.linspace(y1+0j,y2+0j,npoints)
    for c in prange(0,npoints*npoints):#easier to parrelelize 1 loop
            flatstabilities[c]=iterator(f,maxdepth,xvals[c//npoints]+yvals[c%npoints],D)#// and % for turning c into x y
    stabilities=flatstabilities.reshape((npoints,npoints))       
    return stabilities,extent

def covfunc1thread(f,npoints,maxdepth,iterator,fractgenerator,iscomplex=True):
    def cov(x1,x2,y1,y2):#by wrapping the function it remebers f, npoints and other params meaning they 
        #dont need to be passed into any context that calls the function
        return fractgenerator(f,x1,x2,y1,y2,npoints, maxdepth,iscomplex,iterator=iterator)
    return cov 


"""wraps lamdified sympy fuyntions so that they can be pickled
def wrapperforpickle(func):
    def wrapped(x):
        return func(x)
    return wrapped"""

"""default values are all reasonable, should only need to change whats specific for use case"""
def drawStabilityFractal(x1=-2.0,x2=2.0,y1=-2.0,y2=2.0,fractgenerator=genfractfromvertexfornjit
    ,iterator=iterator,f=mandlebrot,npoints=1000, maxdepth=400,plottype="imshow",cmap="Dark2"
    ,ncycles=8,divlim=2,divergencedetector=absdivergencedetectoroptimised,cycledetector=cycledetector):
    @njit
    def prepiterator(function, maxdepth, value,D):#adds settings to iterator.
        return iterator(function, maxdepth, value,D,divergencedetector=divergencedetector,cycledetector=cycledetector,cycles=ncycles,divlim=divlim)
    stabilities,startbounds=fractgenerator(f,x1,x2,y1,y2,npoints=npoints,maxdepth=maxdepth,iterator=prepiterator)#gens first view
    cov = covfunc1thread(f=f,npoints=res,maxdepth = maxdepth,iterator=prepiterator,fractgenerator=fractgenerator)#integrates options into gen
    GUI(stabilities,startbounds,generator=cov,plottype=plottype,cmap=cmap)#draws view and incorperates gen, for zooming

"""default values are all reasonable, should only need to change whats specific for use case 
f can be given as a sympy expression, string that parses to sympy, or calleable function pared with fprime"""
def drawnewtontypefractal(x1=-2.0,x2=2.0,y1=-2.0,y2=2.0,fractgenerator=newtonsfractalfornjit
    ,iterator=newtonsmethod,f="x**3+1+x**5",fprime=None,npoints=1000, maxdepth=200,tol=1e-16
    ,plottype="imshow",cmap="Dark2",ftype="Opencl"):
    
    if type(f)==type("string"):
        f=parse_expr(f)
    if isinstance(f,sp.Basic):#checks if sympy expression, need 2 types as different types of sympy expressions
        fprime = sp.diff(f,x)
        f=sp.lambdify(x,f,"numpy")
        fjit=njit(nopython=True,fastmath=True,locals={"x":complex128})(f)
        fprime=sp.lambdify(x,fprime,"numpy")
        fprimejit=njit(nopython=True,fastmath=True,locals={"x":complex128})(fprime)        
    else:
        fjit=vectorize(nopython=True,fastmath=True,locals={"x":complex128})(f)
        fprimejit=vectorize(nopython=True,fastmath=True,locals={"x":complex128})(fprime)
    assert callable(fjit)
    assert callable(fprimejit)
    if tol==None:
        tol=(((x1-x2)/npoints)**2+((y1-y2)/npoints)**2)/4
    
    @njit
    def prepiterator(X0,N,tol):
        return iterator(fjit,fprimejit,X0,tol=tol,N=N,maxdepth=maxdepth)
    def prepfractgen(x1,x2,y1,y2):
        return fractgenerator(x1,x2,y1,y2,npoints=npoints, maxdepth=maxdepth,tol=tol,iterator=prepiterator)
    stabilities,startbounds=fractgenerator(x1,x2,y1,y2,npoints=npoints,maxdepth=maxdepth,iterator=prepiterator) 
    GUI(stabilities,startbounds,generator=prepfractgen,plottype=plottype,cmap=cmap)#draws view and incorperates gen, for zooming
