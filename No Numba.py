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
from numba import njit
np.warnings.filterwarnings("ignore")

plt.ion()


"""To do:
fix zoom with newtontype fractals atm some kind of rendering issue or sommin
add second oder newton method fractal
add line fractals such as sepinski triangle"""





def generalcomplexpoly(Zn,d,a=2,b=2,power=2):
    #returns value of complex polynomial of the form z^power+a*Zn+b*j+d
    return Zn**power+a*Zn+b*1j+d
@njit("complex128(complex128,complex128)",fastmath=True)
def mandlebrot(Zn,C):
    pow=2
    return Zn**pow+C



def absdivergencedetectoroptimised(ys,val=2):
 if abs(ys)>val:
     return True




def cycledetector(ys,D,val=2):
    #if abs(ys[2]-ys[0])<1e-4:
    #if math.isclose(abs(ys[2]),abs(ys[0]),rel_tol=1e-6):
    D=D*(1-1e-2)
    if (abs(ys[2]-ys[0])<D):
        return True,-2
    if (abs(ys[3]-ys[0])<D):
        return True,-3
        
    return False,0



def newtonsmethod(f,fprime,x0,tol=1e-6,maxdepth=500):
    x=x0
    for i in range(maxdepth):
        x=x-f(x)/fprime(x)
        if abs(f(x))<tol:
            if not x.imag==0:#if not a pure real number
                x=x.real+x.imag#convert to real number that is different from its complex conjugate
                #preferably as different as possible from the other roots
                #this is required as matplotlib canot colourise complex numbers
            return round(x,5)
            #return (round(x,5)+round(x.imag,5)*1j)
    x=x.real+x.imag
    return round(x,5)



   
  
def newtonsfractal(f,x1,x2,y1,y2,npoints=600, maxdepth=80,tol=1e-6,iscomplex=True,iterator=newtonsmethod):
    x = sp.symbols('x')
    dydx= sp.diff(f,x)
    dydx=sp.lambdify(x,dydx)
    f=sp.lambdify(x,f)
    xvals = np.linspace(x1,x2,npoints)
    yvals = np.linspace(y1,y2,npoints)
    print(npoints)
    print((npoints,npoints))
    roots = np.zeros((npoints,npoints))#cannot be dtype complex as it will break the colouring
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
    

"""function should be callable, in form of lamdified sympy
divergence detector should return True if diverged"""

def iterator(function, maxdepth, value,D,divergencedetector=absdivergencedetectoroptimised,cycledetector=cycledetector):
    

    ys=[function(0,value),
    function(function(0,value),value),
    function(function(function(0,value),value),value),
    function(function(function(function(0,value),value),value),value)]
    
    #first 4 function calls
    #3 points allow 2 differences for comparing
    for i in range(maxdepth-3):
        cycle,period=cycledetector(ys,D)
        if cycle:
            return period
            return maxdepth*1.5
        if divergencedetector(ys[-1]):
            return i
        #if abs(ys[-1])>2:
            #return i

        """ys.appendleft(function(ys[-1],value))
        ys.pop()   """                     
        ys[-1]= function(ys[-1],value)
        ys[0]=ys[1]
        ys[1]=ys[2]
        ys[2]=ys[3]
    return maxdepth*1.5

"""Takes complex to complex function, and complex value, 
and returns reecursion count before value > lim"""
#@njit
def simprecurse(function, maxdepth, value,D,lim=16):
    ys=function(0,value)
    for i in range(maxdepth-1):
        if abs(ys)>lim: 
            return i                        
        ys=function(ys,value)
    return maxdepth*1.5


   


def genfractfromvertex(function,x1,x2,y1,y2,npoints=500, maxdepth=150,iscomplex=True,iterator=simprecurse):
    #print("Generating fractal from vertex")
    
    
    stabilities = np.zeros((npoints,npoints))
    #print(stabilities.shape)
    dx=(x1-x2)/npoints
    dy=(y1-y2)/npoints
    D=math.sqrt(dx**2+dy**2)
    #extent = [x1-dx,x2+dx,y1-dy,y2+dy]
    extent = [x1+dx,x2-dx,y1+dy,y2-dy]
    #print(extent)
    xvals = np.linspace(x1,x2,npoints)
    yvals = np.linspace(y1,y2,npoints)
    if iscomplex:
        xvals=xvals*1j
        xc=0
        for x in xvals:
            yc=0
            for y in yvals:
                stabilities[xc,yc]=iterator(function,maxdepth,(x+y),D)#complex numbers must be combined
                yc+=1
            xc+=1
    else:
        
        xc=0
        for x in xvals:
            yc=0
            for y in xvals:
                stabilities[xc,yc]=iterator(function,maxdepth,(x,y),D)
                yc+=1
            xc+=1
    #print("finished stabilities")
    #print(extent)
    return stabilities,extent



"""Does the same as genfractfromvertex, but is compatible with @njit"""
#@njit
def genfractfromvertexfornjit(f,x1,x2,y1,y2,npoints=500, maxdepth=150,iscomplex=True,iterator=simprecurse):
    #print("Generating fractal from vertex")
    lim=2
    stabilities = np.zeros((npoints,npoints),dtype=np.float64)
    #print(stabilities.shape)
    dx=(x1-x2)/npoints
    dy=(y1-y2)/npoints
    D=math.sqrt(dx**2+dy**2)
    #extent = [x1-dx,x2+dx,y1-dy,y2+dy]
    extent = [x1+dx,x2-dx,y1+dy,y2-dy]
    #print(extent)

    xvals = np.linspace(x1*1j,x2*1j,npoints)
    yvals = np.linspace(x1,x2,npoints)
    
    xc=0
    for x in xvals:
        yc=0
        for y in yvals:
            stabilities[xc,yc]=iterator(f,maxdepth,(x+y),D,lim)#complex numbers must be combined
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





"""splits domain defined by x1,x2,y1,y2 into n subdomains, completely covering the domain. Outputs subdomains as a list of (x1,x2,y1,y2) tuples"""
def splitdomain(x1,x2,y1,y2,n):
    n=int(math.sqrt(n))
    print("\n\n  splitting domain into",n**2,"subdomains \n\n")
    dx=abs((x2-x1)/n)
    dy=abs((y2-y1)/n)
    return [(x1+i*dx,x1+(i+1)*dx,y1+j*dy,y1+(j+1)*dy) for i in range(n) for j in range(n)]
    #return [(x1+i*dx,x1+(i+1)*dx,y1+i*dy,y1+(i+1)*dy) for i in range(n)]

    
"""Sort rows by x1 and collumns by y1"""
def reconectdomains(domains):
    domains.sort(key=lambda x: x[0][1])
    rows=[]
    rowlen = int(math.sqrt(len(domains)))
    #print(np.shape(domains))
    #print(rowlen)
    wholedomain=[]
    rowsandcols=np.reshape(domains,(rowlen,rowlen,2))
    for i in range(rowlen):
        newrow=np.array(rowsandcols[i][0][0])
        for j in range(1,rowlen):
            #print(rowsandcols[i][j][0])
            newrow= np.hstack((newrow,rowsandcols[i][j][0]))
        if i==0:
            wholedomain=newrow
        else:
            wholedomain=np.vstack((wholedomain,newrow))
    return wholedomain

"""Splits domain into nparts, and generates a fractal for each part
fract generator is a function that takes a function, domain, and npoints and returns a fractal examples inclue genfractfromvertex and newtonsfractal
iterator is combined with a fractal generator a iterator is required for each generator, and controls what type of iteration is used, examples include 
newtons method and other root finders for newtons fractal, and iterator"""
def multithreadgen(function,x1,x2,y1,y2,threads=9,npoints=600, maxdepth=200,complex=True,iterator=simprecurse,fractgenerator=genfractfromvertexfornjit):
    print(npoints)
    extent = [x1,x2,y1,y2]
    print("\n\n\n\n")
    print(extent)
    print("\n\n\n\n")
    splitdomains=splitdomain(x1,x2,y1,y2,threads)
    threads = len(splitdomains)
    pool = Pool(processes=threads)
    resultssplit=[]
    for i in range(threads):
        thread = pool.apply_async(fractgenerator,args=(function,*splitdomains[i]),kwds={"npoints":int(npoints/threads),"maxdepth":maxdepth,"iscomplex":complex,"iterator":iterator})
        resultssplit.append([thread.get()])
    results = reconectdomains(resultssplit)
    return results,extent
    
def covfunc(multithreadgen,npoints,maxdepth,iterator,fractgenerator):
    def cov(function,x1,x2,y1,y2,npoints=npoints, maxdepth=maxdepth,complex=True,threads=9):
        
        return multithreadgen(function,x1,x2,y1,y2,threads,npoints, maxdepth,complex,iterator=iterator,fractgenerator=fractgenerator)
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
def Draw(stabilities,extent,func,generator = multithreadgen,plottype="imshow",cmap="tab10",Grid=False,plotfunc=plotfract):
    
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
   res = 500
   maxdepth = 50
   fractgenerator=genfractfromvertex#fornjit
   #f=np.e**x+0j#example of seed for Newton fractal
   #f=sp.lambdify(x,f,'numpy')
   f=mandlebrot
   #stabilities,startbounds = multithreadgen(mandlebrot,-2,2,-2,2)#x y axis switched somwhere
   #stabilities,startbounds = multithreadgen(f,-0.6,0.6,-0.4,0.4,iterator=newtonsmethod,fractgenerator=newtonsfractal)#x y axis switched somwhere
   
   #stabilities,startbounds= newtonsfractal(f,-4,4,-4,4)
   
   starttime = timeit.default_timer()
   stabilities,startbounds=multithreadgen(f,-2,2,-2,2,npoints=res,maxdepth=maxdepth,iterator=iterator,fractgenerator=fractgenerator)
   #stabilities,startbounds=newtonsfractal(f,-0.6,0.6,-0.4,0.4,npoints=2000)
   print("First run:", timeit.default_timer() - starttime)
   cov = covfunc(multithreadgen,res,maxdepth = maxdepth,iterator=simprecurse,fractgenerator=fractgenerator)
   Draw(stabilities,startbounds,f,generator=cov,plottype="imshow",cmap="Pastel1")
   