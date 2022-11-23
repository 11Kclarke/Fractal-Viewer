import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from matplotlib.backends import backend_tkagg as tkagg
import sys
import timeit
from matplotlib.widgets import TextBox
#plt.ion()
def plotfract(cmap,stabilities=None,extent=None,plottype=None,ax=None):    
        print(f"{extent} from inside plotting")
        if plottype=="imshow":
            minval=np.min(stabilities)
            maxval=np.max(stabilities)
            if maxval-minval>1000 and minval<0:
                if minval>0:minval=-0.01
                if maxval<0:maxval=0.01
                maxval*=1/30
                #minval=-np.log(-minval)
                minval=minval/400
                divnorm = mpl.colors.TwoSlopeNorm(vmin=minval, vcenter=0, vmax=maxval)
                ax.imshow(stabilities,extent=extent,origin='lower',cmap=cmap,norm=divnorm)
            else:
                ax.imshow(stabilities,extent=extent,origin='lower',cmap=cmap)
            #ax.imshow(stabilities,extent=extent,origin='lower',cmap=cmap)
        elif plottype=="contour":
            shape=np.shape(stabilities)
            xcoords= np.linspace(extent[0],extent[1],shape[0])
            ycoords= np.linspace(extent[2],extent[3],shape[1])
            ax.contourf(xcoords,ycoords,stabilities, cmap=cmap)
            #ax.contour(xcoords,ycoords,stabilities)
        elif plottype=="scatter":
            xcoords= np.linspace(extent[0],extent[1],shape[0])
            ycoords= np.linspace(extent[2],extent[3],shape[1])
            ax.scatter(-xcoords,ycoords,c=stabilities[:,2],cmap=cmap)
        else:
            print(plottype,"is not a valid plot type")
        return cmap

def setupfig(stabilities,extent,Grid,plottype,cmap,plotfunc=plotfract):
    mpl.use('TKAgg')
    fig, ax = plt.subplots()
    ax.set_aspect("auto")
    plt.grid(Grid)
    ax.set(title='Press H for help')
    ax.spines.left.set_position('zero')
    ax.spines.right.set_color('none')
    ax.spines.bottom.set_position('zero')
    ax.spines.top.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim(extent[0],extent[1])
    ax.set_ylim(extent[2],extent[3])
    cmap=plotfunc(cmap,stabilities,extent,plottype,ax)
    return fig,ax
#should prolly be a class
def ZoomableFractalViewer(stabilities,extent,generator,args=[],plottype="imshow",cmap="terrain",Grid=False,plotfunc=plotfract,plotorbits=True):
    print(args)
    print(*args)
    fig,ax=setupfig(stabilities,extent,Grid,plottype,cmap)
    try:
        generator,Orbitgenerator=generator
    except TypeError:
        print("Orbit gen functions not passed")
        plotorbits=False
    def on_press(event):
        sys.stdout.flush()
        if event.key == 'h':
            print("\n\n\n\n")
            print("h: help")
            print("q: quit")
            print("Zoom to regen: regen and draw at full resolution")
            print("p: to print information")
            print("g: to toggle grid and axis")
            print("Move mouse while holding ctrl to see path taken from mouse location")
            print("\n\n\n\n")
        if event.key == 'g':
            if Grid:
                ax.set_axis_off()
                Grid=False
            else:
                ax.set_axis_on()
                Grid=True
            ax.grid(Grid)
            ax.xaxis.set_visible(Grid)
            ax.yaxis.set_visible(Grid)
        if event.key=='q':
            plt.close()
            return
        if event.key=='p':
            print("New Axis limits:")
            xlim=ax.get_xlim()
            ylim=ax.get_ylim()
            print(ax.get_xlim())
            print(ax.get_ylim())
            print(f"area of visible {(ax.get_xlim()[0]-ax.get_xlim()[1])*(ax.get_ylim()[0]-ax.get_ylim()[1])}")
            print("\n\n\n\n")
            print("Using generator:"+"\n"+str(generator))
            print("\n\n\n\n")
            print("extent in form to be inputted")
            print(ylim[0],ylim[1],xlim[0],xlim[1])
            print("\n\n\n\n")
            print("in on press cmap is",cmap)
    if plotorbits:
        def mouse_move(event):
            if event.key=="control":
                x, y = event.xdata, event.ydata
                if x and y:
                    fig.canvas.flush_events()
                    #print("postflush")
                    #print(fig.canvas.events)
                    #origtime=timeit.default_timer()
                    
                    #plt.pause(0.5)
                    Orbit=Orbitgenerator(x,y,*args)
                    #print(f"orbits genned in {timeit.default_timer()-origtime}")
                    #time=timeit.default_timer()
                    for i in ax.lines:
                        ax.lines.remove(i)
                    
                    #print(f"old lines removed in {timeit.default_timer()-time}")
                    ax.plot(np.real(Orbit),np.imag(Orbit),"bo-")
                    plt.draw()
                    
                    #print(f"Full Orbit genned and plotted in {timeit.default_timer()-origtime}")
                    #fig.canvas.flush_events()
        fig.canvas.mpl_connect('motion_notify_event', mouse_move)
    def resize(event):
        nonlocal cmap#ewwwww
        x= ax.get_xlim()
        y= ax.get_ylim()
        if extent!=[*x,*y]:
            #extent[:]=*x,*y
            stabilities,extent[:]=generator(y[0],y[1],x[0],x[1],*args)
            cmap=plotfunc(cmap,stabilities,[*x,*y],plottype,ax)
    fig.canvas.mpl_connect('button_release_event',resize)
    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.show(block=True)
    """def onsubmit(event):
        #need to have none local (global) as cannot return a value from this function
        #by sharing a variable between functions it will be changed in the on pressfunction
        nonlocal cmap
        cmap = plotfunc(event,stabilities,extent,plottype,ax)
        
        

    axbox = fig.add_axes([0.1, 0.05, 0.8, 0.075])
    text_box = TextBox(axbox, "Colour map:", textalignment="center")
    text_box.on_submit(onsubmit)
    text_box.set_val(cmap)  # Trigger `submit` with the initial string.
    """
    
def StabilityandJuliaDoubleplot(Stabilities,JuliaVals,extent,JuliaFractalConst,generator,juliagenerator,plottype="imshow",cmap="terrain",extent2=None,args=[],plotfunc=plotfract,plotorbits=True):
    if not extent2:extent2=extent
    extent2orig=extent2.copy()
    from matplotlib import gridspec
    MouseCurrentaxis=None
    plt.ion()
    try:
        generator,StabilityOrbitgenerator=generator
        juliagenerator,juliaOrbitgenerator=juliagenerator
    except TypeError:
        print("Orbit gen functions not passed")
        plotorbits=False
    fig=plt.figure()
    fig.set_size_inches(12,6)
    fig.set_label("Attractor Explorer")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1],
         wspace=0.2, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)
     
    #fig, ax = plt.subplots(2)
    ax=[plt.subplot(gs[0,0]),plt.subplot(gs[0,1])]
    
    #plt.tight_layout()
    GridToggle=True
    ax[0].set_title("Stability of Julia when C is pixelvalue")
    ax[1].set_title("Julia of input Function C = "+str(JuliaFractalConst))
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].spines['left'].set_position(('zero'))
    ax[0].spines['bottom'].set_position('zero')
    
    
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax[1].spines['left'].set_position('center')
    ax[1].spines['bottom'].set_position('center')
    ax[0].spines['left'].set_position('center')
    ax[0].spines['bottom'].set_position('center')
    # Eliminate upper and right axes
    ax[1].spines['right'].set_color('none')
    ax[1].spines['top'].set_color('none')
    ax[0].spines['right'].set_color('none')
    ax[0].spines['top'].set_color('none')
    # Show ticks in the left and lower axes only
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].yaxis.set_ticks_position('left')
    ax[0].xaxis.set_ticks_position('bottom')
    ax[0].yaxis.set_ticks_position('left')
    #extent,Stabilities=generator(*extent,*args)
    cmap=plotfunc(cmap,Stabilities,extent,plottype,ax[0])
    #extent2,JuliaVals=generator(*extent2,c,*args)
    cmap=plotfunc(cmap,JuliaVals,extent2,plottype,ax[1])
    plt.draw()                
    
    
    
    def onclick(event):
        nonlocal MouseCurrentaxis
        if event.key == "control" and MouseCurrentaxis==0:
            nonlocal cmap
            nonlocal JuliaFractalConst
            x,y=event.xdata,event.ydata
            ax[1].set_title("Orbit of input Function X0,Y0 = "+str(complex(round(x,4),round(y,4))))
            JuliaFractalConst=complex(x,y)
            extent2[:]=extent2orig[:]
            #print(f"Xlim Ylim pre change {(xlim,ylim)}")
            ax[1].set_xlim(extent2[0],extent2[1])
            ax[1].set_ylim(extent2[2],extent2[3])     
            JuliaVals,notextent2=juliagenerator(*extent2,JuliaFractalConst,*args)
            cmap=plotfunc(cmap,JuliaVals,extent2,plottype,ax[1])                           
            ax[1].xaxis.set_ticks_position('bottom')
            ax[1].yaxis.set_ticks_position('left')
            ax[1].spines['right'].set_color('none')
            ax[1].spines['top'].set_color('none')
            ax[1].spines['left'].set_position('center')
            ax[1].spines['bottom'].set_position('center')
            plt.draw()
        else:
            print("Control click to generate new Julia Fractal originating from mouse coords")
    def resize(event):
        nonlocal cmap#ewwwww
        nonlocal MouseCurrentaxis
        xlim= ax[0].get_xlim()
        ylim= ax[0].get_ylim()
        if extent!=[*xlim,*ylim] and MouseCurrentaxis==0:
            #print("zooming image 1")
            stabilities,extent[:]=generator(ylim[0],ylim[1],xlim[0],xlim[1],*args)
            cmap=plotfunc(cmap,stabilities,[*xlim,*ylim],plottype,ax[0])
        else:
            xlim= ax[1].get_xlim()
            ylim= ax[1].get_ylim()
            if extent2!=[*xlim,*ylim] and MouseCurrentaxis==1:
                #print("zooming image 2")
                stabilities,extent2[:]=juliagenerator(ylim[0],ylim[1],xlim[0],xlim[1],JuliaFractalConst,*args)
                cmap=plotfunc(cmap,stabilities,[*xlim,*ylim],plottype,ax[1])
    
        
    def on_enter_axes(event):
        nonlocal MouseCurrentaxis
        if event.inaxes==ax[0]:
            #print("entered axis 0")
            MouseCurrentaxis=0
        if event.inaxes==ax[1]:
            MouseCurrentaxis=1
            #print("entered axis 1")
    def on_leave_axes(event):
        nonlocal MouseCurrentaxis
        #print(f"leaving axis {MouseCurrentaxis}")
        MouseCurrentaxis=None
    
    def on_press(event):
        sys.stdout.flush()
        if event.key == 'h':
            print("\n\n\n\n")
            print("h: help")
            print("q: quit")
            print("Zoom to regen: regen and draw at full resolution")
            print("p: to print information")
            print("g: to toggle grid and axis")
            print("Move mouse while holding ctrl to see path taken from mouse location")
            print("\n\n\n\n")
        if event.key == 'g':
            nonlocal GridToggle
            if GridToggle:
                GridToggle=False
                [axi.set_axis_off() for axi in ax]
            else:
                GridToggle=True
                [axi.set_axis_on() for axi in ax]
            for axi in ax:
                axi.grid(GridToggle)
                axi.axis(GridToggle)
                axi.xaxis.set_visible(GridToggle)
                axi.yaxis.set_visible(GridToggle)
            plt.draw()
            
        if event.key=='q':
            plt.close()
            return
        if event.key=='p':
            print("New Axis limits:")
            for axi in ax:
                print(axi.get_xlim())
                print(axi.get_ylim())
            print("\n\n\n\n")
            print("Using generator:"+"\n"+str(generator))
            print("cmap is",cmap)
    
    if plotorbits:
        def mouse_move(event):
            if event.key=="control":
                if MouseCurrentaxis ==0:
                    argshere=args.copy()
                    Orbitgenerator=StabilityOrbitgenerator
                    axs=ax[0]
                elif MouseCurrentaxis ==1:
                    argshere=[JuliaFractalConst]+args
                    Orbitgenerator=juliaOrbitgenerator
                    axs=ax[1]
                else:
                    return  
                x, y = event.xdata, event.ydata
                if x and y:# and xpix%2==0:
                    sys.stdout.flush()
                    fig.canvas.flush_events()
                    Orbit=Orbitgenerator(x,y,*argshere)
                    for i in axs.lines:
                        axs.lines.remove(i)
                    axslims=(*axs.get_xlim(),*axs.get_ylim())
                    axs.plot(np.real(Orbit),np.imag(Orbit),"bo-")
                    axs.set_xlim(axslims[0],axslims[1])
                    axs.set_ylim(axslims[2],axslims[3])
                    plt.draw()
                    sys.stdout.flush()
        fig.canvas.mpl_connect('motion_notify_event', mouse_move)
    
        
    fig.canvas.mpl_connect('button_release_event',resize)
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', resize)
    fig.canvas.mpl_connect('axes_enter_event', on_enter_axes)
    fig.canvas.mpl_connect('axes_leave_event', on_leave_axes)
    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.show(block=True)    
