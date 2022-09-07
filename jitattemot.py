from tokenize import Exponent
from matplotlib.transforms import Transform
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr
from multiprocessing import Pool
import sys
import timeit
import math
from matplotlib.widgets import TextBox
import numba
from numba import *
import inspect
np.warnings.filterwarnings("ignore")

plt.ion()


"""To do:
fix zoom with newtontype fractals atm some kind of rendering issue or sommin
add second oder newton method fractal
add line fractals such as sepinski triangle"""




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

@njit("complex128(complex128,complex128)",fastmath=True)
def exponential(Zn,C):
    pow=2.718
    return pow**Zn+C

@njit("complex128(complex128,complex128)",fastmath=True)
def TriginTrig(Zn,C):
    return np.cos(C+np.sin(Zn))


#very cool, but needs to start zoomed
@njit("complex128(complex128,complex128)",fastmath=True)
def reciprocalmandlebrot(Zn,C):
    return 1/(Zn**2+C)



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
def newtonsmethod(f,fprime,x0,tol=1e-6,N=4,maxdepth=500):#,f=None,fprime=None):
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
    return -1 
"""
@vectorize(nopython=True,fastmath = True)
def vectnewtonsmethod(x0,tol=1e-6,N=4,maxdepth=500):#,f=None,fprime=None):
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
            return np.float64(root)
    return np.float64(-1) 
"""
   
 

@njit("complex128(complex128)",fastmath=True)
def poly(x):
    return x**3+2*x**2+1j*x-10
@njit("complex128(complex128)",fastmath=True)
def polyderiv(x):
    return 3*x**2+4*x+1j

@njit(fastmath = True,parallel=True,nogil = True)
def newtonsfractalfornjit(x1,x2,y1,y2,npoints=600, maxdepth=80,tol=1e-6,iterator=newtonsmethod,Nroots=None):#,f=None,dydx=None):
    #print(type(tol))
    #print(str(tol))
    tol=(((x1-x2)/npoints)**2+((y1-y2)/npoints)**2)/4
    N=int(np.log10(tol))
    flatroots = np.ones((npoints,npoints),dtype=np.float64).flatten()#cannot be dtype complex as it will break the colouringa
    extent = [x1,x2,y1,y2]
    xvals = np.linspace(x1*1j,x2*1j,npoints)
    yvals = np.linspace(y1+0j,y2+0j,npoints)
    
    #put in 1 loop as it parralelizes easier
    for c in prange(0,npoints*npoints):
            flatroots[c]=iterator(xvals[c//npoints]+yvals[c%npoints],Tol=tol,N=N)
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





"""returns function that returns fractal defined by f, in bounds of x1,y2"""
def factoryfunc(f,gen = newtonsfractalfornjit, iterator=None,npoints=600, maxdepth=400,tol=1e-9,iscomplex=True,Nroots=None):
    fprime = sp.diff(f,x)
    f=sp.lambdify(x,f,"numpy")
    print(inspect.getsource(f))
    fjit=njit(nopython=True,locals={"x":complex128})(f)
    fprime=sp.lambdify(x,fprime,"numpy")
    print(inspect.getsource(fprime))
    fprimejit=njit(nopython=True,locals={"x":complex128})(fprime)
    """Nroots=None
    if Nroots == None and iterator == None:
        iterator = vectnewtonsmethod
    elif Nroots != None and iterator == None:
        iterator = newtonsmethodknownrootcount
    else:
        print(iterator)
        print("has been selected manually")"""
    print(tol)
    #def vectiterator(x0,f=fjit,dydx=fprimejit):
        #return iterator(x0,f=fjit,dydx=fprimejit)
    #@vectorize("complex128(float64)",nopython=True,fastmath = True)
    #def vectorizeiterator(X0):
    
    print(fjit)
    print(fprimejit)
    print(fjit(fjit(1j)))
    print(fprimejit(fprime(1j)))
    N=int(np.log10(tol))
    @njit(float64(complex128),nopython=True,locals={"x":complex128,"root":float64})
    def vectnewtonsmethod(x0):#,f=None,fprime=None):
        x=x0+0j
        root=np.float64(-1)
        for i in range(maxdepth):
            x=x-fjit(x)/fprimejit(x)
            if abs(fjit(x))<tol:
                root=np.float64(round(abs(x.real+x.imag),-(N+2)))#convert to real number that is different from its complex conjugate
                #root = np.float64(int(abs(x.real+x.imag)*10**-(N+1))*10**-(N+1))
                #preferably as different as possible from the other roots
                #this is required as matplotlib canot colourise complex numbers
                break
        return root
        #return vectnewtonsmethod(X0)
    print(vectnewtonsmethod)
    def newtonsfractal(x1,x2,y1,y2,npoints=npoints, maxdepth=maxdepth,tol=tol,iterator=vectnewtonsmethod,Nroots=Nroots):
        print(iterator)
        return gen(x1,x2,y1,y2,npoints=npoints, maxdepth=maxdepth,tol=tol,iterator=iterator,Nroots=Nroots) 
    return newtonsfractal
    

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

def plotfract(cmap,stabilities=None,extent=None,plottype=None,ax=None,shape=None):    
        if plottype=="imshow":
            ax.imshow(stabilities,extent=extent,origin='lower',cmap=cmap)
        elif plottype=="contour":
            xcoords= np.linspace(extent[0],extent[1],shape[0])
            ycoords= np.linspace(extent[2],extent[3],shape[1])
            ax.contourf(xcoords,ycoords,stabilities, cmap=cmap)
            #ax.contour(xcoords,ycoords,stabilities)
        elif plottype=="scatter":
            print(len(stabilities))
            
            xcoords= np.linspace(extent[0],extent[1],shape[0])
            ycoords= np.linspace(extent[2],extent[3],shape[1])
            print(len(xcoords))
            print(len(ycoords))

            ax.scatter(-xcoords,ycoords,c=stabilities[:,2],cmap=cmap)
        else:
            print(plottype,"is not a valid plot type")
        return cmap

#genfractfromvertex(func,0,1,0,1,npoints=500, maxdepth=150,complex=True,recurse=recurse)
def Draw(stabilities,extent,generator = genfractfromvertexfornjit,plottype="imshow",cmap="tab10",Grid=False,plotfunc=plotfract):
    
    fig, ax = plt.subplots()
    ax.set_aspect("auto")
    plt.grid(Grid)
    ax.set(title='Press H for help')
    ax.set_xlim(extent[0],extent[1])
    ax.set_ylim(extent[2],extent[3])
    ax.spines['left'].set_position(('zero'))
    ax.spines['bottom'].set_position('zero')
    shape=np.shape(stabilities)
    #over use of kwargs so it can be easily caleable from buttom press
    #ax.imshow(stabilities,extent=extent,origin='lower')
    print(cmap)
    cmap=plotfunc(cmap,stabilities,extent,plottype,ax,shape)
    print(cmap)
    def on_press(event):
        nonlocal cmap
        sys.stdout.flush()
        #print(event.key)
        if event.key == 'h':
            print("\n\n\n\n")
            print("h: help")
            print("q: quit")
            print("m: regen and draw at full resolution  (r is already taken by matplotlib as reset)")
            print("p: to print information")
            print("g: to toggle grid")
            print("\n\n\n\n")
        if event.key == 'g':
            if Grid:
                Grid=False
            else:
                Grid=True
            ax.grid(Grid)
        if event.key=='q':
            plt.close()
            return
        if event.key=='p':
            print("New Axis limits:")
            print(ax.get_xlim())
            print(ax.get_ylim())
            print("\n\n\n\n")
            print("Using generator:"+"\n"+str(generator))
            print("\n\n\n\n")
            #print("Using seed function:"+"\n"+str(func))
            print("\n\n\n\n")
            print("in on press cmap is",cmap)
        if event.key=='m':
            print("redrawing")
            print("New Axis limits:")
            print(ax.get_xlim())
            print(ax.get_ylim())
            
            xlim=ax.get_xlim()
            ylim=ax.get_ylim()
            
            ax.spines['left'].set_position(('zero'))
            ax.spines['bottom'].set_position('zero')
            
            #stabilities,extent=generator(func,xlim[0],xlim[1],ylim[0],ylim[1])
            starttime = timeit.default_timer()
            stabilities,extent=generator(ylim[0],ylim[1],xlim[0],xlim[1])#,npoints=npoints)
            print("Time To Redraw:", timeit.default_timer() - starttime)

            """somwhere in the generator function the x and y are switched i cant find where but this switches them back its not a good solution"""
            extent=np.array([extent[2],extent[3],extent[0],extent[1]])
            """This line switches the x and y coordinates back"""


            cmap=plotfunc(cmap,stabilities,extent,plottype,ax,shape)
            plt.draw()
            
        #print("back in draw")

    
    cid=fig.canvas.mpl_connect('key_press_event', on_press)
    def onsubmit(event):
        #need to have none local (global) as cannot return a value from this function
        #by sharing a variable between functions it will be changed in the on pressfunction
        nonlocal cmap
        cmap = plotfunc(event,stabilities,extent,plottype,ax,shape)
        
        

    axbox = fig.add_axes([0.1, 0.05, 0.8, 0.075])
    text_box = TextBox(axbox, "Colour map:", textalignment="center")
    text_box.on_submit(onsubmit)
    text_box.set_val(cmap)  # Trigger `submit` with the initial string.

    
    
    
    plt.show(block=True)    

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
    Draw(stabilities,startbounds,generator=cov,plottype=plottype,cmap=cmap)#draws view and incorperates gen, for zooming

"""default values are all reasonable, should only need to change whats specific for use case 
f can be given as a sympy expression, string that parses to sympy, or calleable function pared with fprime"""
def drawnewtontypefractal(x1=-2.0,x2=2.0,y1=-2.0,y2=2.0,fractgenerator=newtonsfractalfornjit
    ,iterator=newtonsmethod,f="x**2-x+1+x**5",fprime=None,npoints=1000, maxdepth=200,plottype="imshow",cmap="Dark2"):
    if type(f)==type("string"):
        f=parse_expr(f)
    if str(type(type(f)))=="<class 'sympy.core.assumptions.ManagedProperties'>":#checks if sympy expression, need 2 types as different types of sympy expressions
        fprime = sp.diff(f,x)
        f=sp.lambdify(x,f,"numpy")
        fjit=njit(nopython=True,fastmath=True,locals={"x":complex128})(f)
        fprime=sp.lambdify(x,fprime,"numpy")
        fprimejit=njit(nopython=True,fastmath=True,locals={"x":complex128})(fprime)
    else:
        fjit=njit(nopython=True,fastmath=True,locals={"x":complex128})(f)
        fprimejit=njit(nopython=True,fastmath=True,locals={"x":complex128})(fprime)
    assert callable(fjit)
    assert callable(fprimejit)
    tol=(((x1-x2)/npoints)**2+((y1-y2)/npoints)**2)/4
    @njit
    def prepiterator(X0,N,Tol):
        return iterator(fjit,fprimejit,X0,tol=Tol,N=N,maxdepth=maxdepth)
    def prepfractgen(x1,x2,y1,y2):
        return fractgenerator(x1,x2,y1,y2,npoints=npoints, maxdepth=maxdepth,tol=tol,iterator=prepiterator)
    stabilities,startbounds=fractgenerator(x1,x2,y1,y2,npoints=npoints,maxdepth=maxdepth,iterator=prepiterator) 
    Draw(stabilities,startbounds,generator=prepfractgen,plottype=plottype,cmap=cmap)#draws view and incorperates gen, for zooming

"""ReadMe:
Easiest way to use is the 2 functions directly above Drawnewtontypefractal and draw stabilitytypefractal. you can simply run these with there defaults
To Gen fractal, you need a generator functionl/iterator, 
that defines the type, for mandlebrot like fractals use genfromvertex, 
for newton types use newtons fractal. 
A generator needs a seed function to be iterated by the generator. For mandlebrot esque, 
use calleable complex function. Later should take sympy equation. Newton types already use sympy.

Draw draws a fractal. For draw to be able to Redraw after zoom it needs to be passed pairing
of gen and seed, as well as the original image.

covfunc is a function wrapper that outputs a generator seed pairing, and wraps it together with all options set,
so it can be passed to Draw without draw needing to be passed all of the generation options.
"""

if __name__ == '__main__':
   
    starttime = timeit.default_timer()
    x,y,t = sp.symbols('x,y,t')
    a,b,c,d = sp.symbols('a,b,c,d')
    res = 1000
    maxdepth = 200
   
    starttime = timeit.default_timer()
    

    #drawStabilityFractal(f=exponential,npoints=4000,maxdepth=200,ncycles=64)
    drawnewtontypefractal()
   
    
    #f=x**2-x+1+x**5#example of seed for Newton fractal
    
    