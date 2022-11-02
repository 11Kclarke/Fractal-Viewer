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
import inspect


import PyopenclNewtonsFractal
import PyopenclStabilityFractal
np.warnings.filterwarnings("ignore")

plt.ion()
x,y,t = sp.symbols('x,y,t')
a,b,c,d,C = sp.symbols('a,b,c,d,C')

"""To do:
fix zoom with newtontype fractals atm some kind of rendering issue or sommin
add second oder newton method fractal
add line fractals such as sepinski triangle"""












    


def plotfract(cmap,stabilities=None,extent=None,plottype=None,ax=None,shape=None):    
        if plottype=="imshow":
            minval=np.min(stabilities)
            if minval>0:minval=-0.01
            maxval=np.max(stabilities)
            if maxval<0:maxval=0.01
            maxval*=1/30
            #minval=-np.log(-minval)
            minval=minval/400
            divnorm = mpl.colors.TwoSlopeNorm(vmin=minval, vcenter=0, vmax=maxval)
            ax.imshow(stabilities,extent=extent,origin='lower',cmap=cmap,norm=divnorm)
            #ax.imshow(stabilities,extent=extent,origin='lower',cmap=cmap)
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

#should prolly be a class
def GUI(stabilities,extent,generator,plottype="imshow",cmap="terrain",Grid=False,plotfunc=plotfract):
    
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
    
    cmap=plotfunc(cmap,stabilities,extent,plottype,ax,shape)
    print(cmap)
    def on_press(event):
        nonlocal cmap#ewww
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

def DrawNewtonsFractalOpencl(x1,x2,y1,y2,fl,fprimel,npoints=1000, maxdepth=200,tol=1e-16):
    if isinstance(fl,str):
        fl=parse_expr(fl)
        fp=sp.diff(f)
        fl=sp.lambdify(x,f)
        fprimel=sp.lambdify(x,fp)
    if fprimel==None:
        fp=sp.diff(f)
        fl=sp.lambdify(x,f)
        fprimel=sp.lambdify(x,fp)
    innerwrap = PyopenclNewtonsFractal.WrapperOpenCltoDraw(x1,x2,y1,y2,fl,fprimel,npoints=npoints, maxdepth=maxdepth,tol=tol)
    Roots,extent=innerwrap(x1,x2,y1,y2)
    GUI(Roots,extent,innerwrap)
    
def DrawStabilityFractalOpencl(x1,x2,y1,y2,fl,npoints=1024, maxdepth=3000,cycles=16,cycleacc=None,ittcountcolouring=True):
    if isinstance(fl,str):
        fl=parse_expr(fl)
        fl=sp.lambdify((x,c),fl)
    if isinstance(fl,sp.Basic):
        fl=sp.lambdify((x,c),fl)
    
    innerwrap = PyopenclStabilityFractal.WrapperOpenCltoDraw(x1,x2,y1,y2,fl,npoints=npoints, maxdepth=maxdepth,cycles=cycles,
                                                             cycleacc=cycleacc,ittCountColouring=ittcountcolouring)
    Roots,extent=innerwrap(x1,x2,y1,y2)
    #Roots=np.log(abs(Roots))
    GUI(Roots,extent,innerwrap)
        
    


"""ReadMe:
Easiest way to use is the 2 functions directly above Drawnewtontypefractal and draw stabilitytypefractal. you can simply run these with there defaults
To Gen fractal, you need a generator functionl/iterator, 
that defines the type, for mandlebrot like fractals use genfromvertex, 
for newton types use newtons fractal. 
A generator needs a seed function to be iterated by the generator. For mandlebrot esque, 
use calleable complex function. Later should take sympy equation. Newton types already use sympy.

Draw draws a fractal. For draw to be able to Redraw after zoom it needs to be passed pairing
of gen and seed, as well as the original image.
"""

if __name__ == '__main__':
   
    starttime = timeit.default_timer()

    res = 1024
    maxdepth = 2048*2
    f=x**2+c
    
    DrawStabilityFractalOpencl(1,-1,1,-1,f,maxdepth=maxdepth,npoints=res,cycles=16,ittcountcolouring=True)
    #DrawNewtonsfractalOpencl(-1,1,-1,1,fl,fpl,npoints=res,maxdepth=500,tol=1e-6)
    
    #drawStabilityFractal(npoints=4000,maxdepth=200,ncycles=8)
    #drawnewtontypefractal(f=f,npoints=1000,x1=-1,x2=1,y1=-1,y2=1)
    
    
    #f=x**2-x+1+x**5#example of seed for Newton fractal
    
    