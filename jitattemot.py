from cmath import isclose
from tokenize import Exponent
from matplotlib.transforms import Transform
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify
from multiprocessing import Pool
import sys
import timeit
import math
from matplotlib.widgets import TextBox
import numba
from numba import njit
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


@njit("Tuple((boolean,int16))(complex128[:],float64,int32)",fastmath=True)
def cycledetector(ys,D,cycles):
    #if abs(ys[2]-ys[0])<1e-4:
    #if math.isclose(abs(ys[2]),abs(ys[0]),rel_tol=1e-6):
    D=D*(1-1e-2)
    for i in range(1,cycles):
        if abs(ys[i]-ys[0])<D:
            return True,-i
    return (False,0)


@njit(fastmath = True)
def newtonsmethod(f,fprime,x0,tol=1e-6,maxdepth=500):
    print(f)
    print(fprime)
    x=x0+0j
    for i in range(maxdepth):
        x=x-f(x)/fprime(x)
        if abs(f(x))<tol:
            root=np.float32(round(abs(x.real+x.imag),5))#convert to real number that is different from its complex conjugate
            #preferably as different as possible from the other roots
            #this is required as matplotlib canot colourise complex numbers
            return root
           


   
 
def newtonsfractal(f,x1,x2,y1,y2,npoints=600, maxdepth=80,tol=1e-6,iscomplex=True,iterator=newtonsmethod):
    x = sp.symbols('x')
    dydx= sp.diff(f,x)
    dydx=sp.lambdify(x,dydx)
    f=sp.lambdify(x,f)
    xvals = np.linspace(x1,x2,npoints)
    yvals = np.linspace(y1,y2,npoints)
    print(npoints)
    print((npoints,npoints))
    roots = np.zeros((npoints,npoints),dtype=np.float32)#cannot be dtype complex as it will break the colouring
    extent = [x1,x2,y1,y2]
    if iscomplex:
        yvals=yvals*1j
    xc=0
    for x in xvals:
        yc=0
        for y in yvals:
            roots[xc,yc]=iterator(f,dydx,x+y,tol=tol,maxdepth=maxdepth)
            yc+=1
        xc+=1
    for i in np.unique(roots):
       print(i)
    return roots,extent



@njit(fastmath = True)
def newtonsfractalfornjit(f,dydx,x1,x2,y1,y2,npoints=600, maxdepth=80,tol=1e-6,iterator=newtonsmethod):#,f=None,dydx=None):
    roots = np.ones((npoints,npoints),dtype=np.float32)#cannot be dtype complex as it will break the colouring
    #stabilities = np.zeros((npoints,npoints),dtype=np.float64)
    extent = [x1,x2,y1,y2]
    xvals = np.linspace(x1*1j,x2*1j,npoints)
    #xvals=xvals*1j
    yvals = np.linspace(y1+0j,y2+0j,npoints)
    print(f)
    print(dydx)
    
    """if iscomplex:
        yvals=yvals*1j"""
    xc=0
    for x in xvals:
        yc=0
        for y in yvals:
            temp=iterator(f,dydx,x+y,tol=tol,maxdepth=maxdepth)
            print(temp)
            print(type(temp))
            roots[xc,yc]=temp
            yc+=1
        xc+=1
    for i in np.unique(roots):
       print(i)
    return roots,extent

"""returns function that returns fractal defined by f, in bounds of x1,y2"""
def factoryfunc(f,x1,x2,y1,y2,gen = newtonsfractalfornjit, iterator=newtonsmethod,npoints=600, maxdepth=80,tol=1e-6,iscomplex=True):
    x = sp.symbols('x')
    dydx= sp.diff(f,x)
    dydx=sp.lambdify(x,dydx,modules="numpy")
    f=sp.lambdify(x,f,modules="numpy")#at this point they should both be normal python functions
    print("------")
    print(f(1+1j))
    print("------")
    f=njit(f,fastmath =True)
    print("------")
    print(f(1+1j))
    print("------")
    dydx=njit(dydx,fastmath =True)
    #both are jit compiles functions by this point
    print(f)
    print(dydx)
    print(iterator)
    print(gen)
    
    def newtonsfractal(placeholder,x1,x2,y1,y2,npoints=npoints, maxdepth=maxdepth,iterator=iterator,tol=1e-6,f=f,dydx=dydx):
        print("------")
        print(f)
        print(dydx)
        print(x1)
        print(f)
        fx = f(1)
        print(fx)
        print(dydx(x1))
        print("------\n\n\n\n\n\n")
        print("REEEEEEEEEEEEE") 
        return njit(gen(x1,x2,y1,y2,npoints=npoints, maxdepth=maxdepth,iterator=iterator,tol=1e-6,f=f,dydx=dydx),fastmath = True) 
    return newtonsfractal
    

"""function should be callable, in form of lamdified sympy
divergence detector should return True if diverged"""
@njit(fastmath = True)
def iterator(function, maxdepth, value,D,divergencedetector=absdivergencedetectoroptimised,cycledetector=cycledetector,divlim=2,cycles=6):
    ys=np.zeros(cycles,dtype=np.complex128)
    ys[0]=value
    for i in range(1,cycles):
        ys[i]=function(ys[i-1],value)
   
    if divergencedetector(ys[-1],divlim):
        return maxdepth
    """
    ys=[function(0,value),
    function(function(0,value),value),
    function(function(function(0,value),value),value),
    function(function(function(function(0,value),value),value),value)]
    """
    #first 4 function calls
    #3 points allow 2 differences for comparing
    for i in range(2,maxdepth-3):
        cycle,period=cycledetector(ys,D,cycles)
        if cycle:
            return period
            return maxdepth*1.5
        if divergencedetector(ys[-1],divlim):
            #if i == 0 or i == 1:
            #   return maxdepth
            return i
        #if abs(ys[-1])>2:
            #return i

        """ys.appendleft(function(ys[-1],value))
        ys.pop()   """
                             
        ys[-1]= function(ys[-1],value)
        for i in range(cycles-1):
            ys[i]=ys[i+1]
        """ys[0]=ys[1]
        ys[1]=ys[2]
        ys[2]=ys[3]
        ys[3]=ys[4]"""
    return maxdepth+10

"""Takes complex to complex function, and complex value, 
and returns reecursion count before value > lim"""
@njit(fastmath = True)
def simprecurse(function, maxdepth, value,D,divergencedetector=absdivergencedetectoroptimised,cycledetector=cycledetector,divlim=2):
    #divergencedetector(0,1)
    
    ys=function(0,value)
    for i in range(maxdepth-1):
        if abs(ys)>divlim: 
            return i                        
        ys=function(ys,value)
    return maxdepth+10


   






"""Does the same as genfractfromvertex, but is compatible with @njit"""
@njit(fastmath=True)
def genfractfromvertexfornjit(f,x1,x2,y1,y2,npoints=600, maxdepth=1000,iscomplex=True,iterator=simprecurse):
    #print("Generating fractal from vertex")
    lim=2
    stabilities = np.ones((npoints,npoints),dtype=np.float64)*maxdepth
    #stabilities = np.zeros((npoints,npoints),dtype=np.float64)
    dx=(x1-x2)/npoints
    dy=(y1-y2)/npoints
    #extent = [x1-dx,x2+dx,y1-dy,y2+dy]
    extent = [x1+dx,x2-dx,y1+dy,y2-dy]
    D=math.sqrt(dx**2+dy**2)
    

    xvals = np.linspace(x1*1j,x2*1j,npoints)
    #xvals=xvals*1j
    yvals = np.linspace(y1+0j,y2+0j,npoints)
    
    xc=0
    for x in xvals:
        yc=0
        for y in yvals:
            #print(absdivergencedetectoroptimised)
            stabilities[xc,yc]=iterator(f,maxdepth,(x+y),D,)
            #divergencedetector=absdivergencedetectoroptimised,cycledetector=cycledetector)
            #complex numbers must be combined
            """value=x+y
            ys=f(0,value)
            for i in range(maxdepth-1):
                if abs(ys)>lim: 
                    stabilities[xc,yc] = i
                    break                        
                ys=f(ys,value)"""
            yc+=1
        xc+=1
            #stabilities[xc,yc] = maxdepth*1.5
    #print(stabilities)
    return stabilities,extent






    
def covfunc(multithreadgen,npoints,maxdepth,iterator,fractgenerator):
    def cov(function,x1,x2,y1,y2,npoints=npoints, maxdepth=maxdepth,complex=True,threads=9):
        
        return multithreadgen(function,x1,x2,y1,y2,threads,npoints, maxdepth,complex,iterator=iterator,fractgenerator=fractgenerator)
    return cov
    
def covfunc1thread(npoints,maxdepth,iterator,fractgenerator):
    def cov(function,x1,x2,y1,y2,npoints=npoints, maxdepth=maxdepth,complex=True):
        
        return fractgenerator(function,x1,x2,y1,y2,npoints, maxdepth,complex,iterator=iterator)
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
def Draw(stabilities,extent,func,generator = genfractfromvertexfornjit,plottype="imshow",cmap="tab10",Grid=False,plotfunc=plotfract):
    
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
            print("Using seed function:"+"\n"+str(func))
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
            stabilities,extent=generator(func,ylim[0],ylim[1],xlim[0],xlim[1])#,npoints=npoints)
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
#@njit    
def exponential(x,y):
    return np.exp(x+y)

"""ReadMe:
To Gen fractal, you need a generator functionl/iterator, 
that defines the type, for mandlebrot like fractals use genfromvertex, 
for newton types use newtons fractal. 
A generator needs a seed function to be iterated by the generator. For mandlebrot esque, 
use calleable complex function. Later should take sympy equation. Newton types already use sympy.

Draw draws a fractal. For draw to be able to Redraw after zoom it needs to be passed pairing
of gen and seed, as well as the original image.

Multithreadgen takes a generator seed pairing, and generates N fratals with N threads then
sticks them back together. This is important to reduce the amount of time it takes to generate

covfunc is a function wrapper that outputs a generator seed pairing, and wraps it together with all options set,
so it can be passed to Draw without draw needing to be passed all of the generation options.
"""
if __name__ == '__main__':
   
    starttime = timeit.default_timer()
    x,y,t = sp.symbols('x,y,t')
    a,b,c,d = sp.symbols('a,b,c,d')
    res = 1500
    maxdepth = 200
   
   
   
   
    #stabilities,startbounds= newtonsfractal(f,-4,4,-4,4)
   
    starttime = timeit.default_timer()
    #stabilities,startbounds=multithreadgen(f,-2,2,-2,2,npoints=res,maxdepth=maxdepth,iterator=iterator,fractgenerator=fractgenerator)
    #(f,-2,2,-2,2,npoints=res,maxdepth=maxdepth,iterator=iterator,fractgenerator=fractgenerator)
    #stabilities,startbounds=genfractfromvertexfornjit(f,-2,2,-2,2,npoints=res,maxdepth=maxdepth,iterator=iterator)
    #stabilities,startbounds=newtonsfractalfornjit(f,-0.6,0.6,-0.4,0.4,npoints=2000)

    
    #for drawing mandlebrot with cycles shown
    fractgenerator=genfractfromvertexfornjit
    iterator = iterator
    f= mandlebrot
    
    """ from sympy import *
    f=x**2-x**3+1j*x-1
    fprime = sp.diff(f,x)
    f=sp.lambdify(x,f,"numpy")
    fjit=numba.njit("complex128(complex128)",f,forceobj=True)
    fprime=sp.lambdify(x,fprime,"numpy")
    fprimejit=numba.njit("complex128(complex128)",fprime,forceobj=True)
    fractgenerator = newtonsfractalfornjit
    #example of seed for Newton fractal
    #fractgenerator = factoryfunc(f,-2.0,2.0,-2.0,2.0)
    iterator=newtonsmethod
    stabilities,startbounds=fractgenerator(f,fprime,-2.0,2.0,-2.0,2.0,npoints=res,maxdepth=maxdepth,iterator=iterator)"""

    
    #stabilities,startbounds=fractgenerator(f,-2.0,2.0,-2.0,2.0,npoints=res,maxdepth=maxdepth,iterator=iterator)
    print("First run:", timeit.default_timer() - starttime)
    cov = covfunc1thread(res,maxdepth = maxdepth,iterator=iterator,fractgenerator=fractgenerator)
    Draw(stabilities,startbounds,f,generator=cov,plottype="imshow",cmap="Dark2")
   