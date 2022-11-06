
import numpy as np
import pyopencl as cl
import pyopencl.array
import pandas as pd
from pyopencl.elementwise import ElementwiseKernel
import datashader as ds
import os
import matplotlib.pyplot as plt
from numba import jit,prange,njit
import timeit

os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
hoppalongx = "k1- Xs[i]"
hoppalongy = "Ys[i]-sign(Xs[i])*( sqrt(fabs(k2*Xs[i]-k3)))"
def prepAttractorCl(f,g,dtype=float):
    from PyToPyOpenCL import subsfunction
    mapclstr=["int ittlim,float k1,float k2, float k3,float **Y,float **X","""
        for (int k=0; k<ittlim; k++){
            resY[i+N*k+1] =__f(X[i+N*k],Y[i+N*k],k1,k2,k3,k4,k5)__;
            resX[i+N*k+1]= __g(X[i+N*k],Y[i+N*k],k1,k2,k3,k4,k5)__;
        }
        """]
    f=("float k1,float k2, float k3,float k4,float k5,float **Y,float **X".split(","),f)
    g=("float k1,float k2, float k3,float k4,float k5,float **Y,float **X".split(","),g)
    mapclstr[1]=subsfunction(f,mapclstr[1],"f",RemoveSemiColon=False)
    mapclstr[1]=subsfunction(g,mapclstr[1],"g",RemoveSemiColon=False)
    print("\n\n")
    print(mapclstr)
    print("\n\n")
    if dtype == np.float64:
            mapclstr[1] = mapclstr[1].replace("float","double")
            mapclstr[1] = mapclstr[1].replace("float","double")
            
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl.create_some_context())
    mapcl = ElementwiseKernel(ctx,*mapclstr,"mapcl")
    return mapcl
            
def iteratefast(Consts, Xs, Ys, resY, resX,mapcl,N,SideLength,precompiled):
    #print("optimised for time")
    Hoppalongiterations=[]
    for i in range(N):
        mapcl(*Consts, Xs, Ys, resY, resX)
        Hoppalongiterations.append([resX.get(),resY.get()])
    return np.array(Hoppalongiterations)

def iterateSavememory(Consts, Xs, Ys, resY, resX,mapcl,N,SideLength,precompiled):
    Hoppalongiterations = np.zeros((N,2,SideLength**2))
    for i in range(N):
        mapcl(*Consts, Xs, Ys, resY, resX)
        Hoppalongiterations[i] =  [resX.get(),resY.get()]
    return Hoppalongiterations

def iterateOpencl(x1,x2,y1,y2,N,mapclstr,SideLength,Consts = np.array([2.0,1.0,0.0],dtype =np.float32),iterator=iteratefast,dtype=np.float32,precompiled=False):
    precompiled=False
    if not precompiled:
        if str(type(mapclstr[0]))=="<class 'pyopencl.elementwise.ElementwiseKernel'>":
            mapcl=mapclstr[0]
            ctx = mapclstr[1]
            queue = mapclstr[2]
            precompiled=True
            print("\n\n Map has been precompiled \n\n")
        elif dtype == np.float64:
            mapclstr[0] = mapclstr[0].replace("float","double")
            mapclstr[1] = mapclstr[1].replace("float","double")
            ctx = cl.create_some_context()
            queue = cl.CommandQueue(ctx)
            mapcl = ElementwiseKernel(ctx,*mapclstr,"mapcl")
        else:
            
            ctx = cl.create_some_context()
            queue = cl.CommandQueue(ctx)
            mapcl = ElementwiseKernel(ctx,*mapclstr,"mapcl")
    
    if SideLength>1:
        xvals = np.linspace(x1,x2,SideLength,dtype =dtype)
        yvals = np.linspace(y1,y2,SideLength,dtype =dtype) 
        resx_np = np.zeros(SideLength**2,dtype =dtype)
        resy_np = np.zeros(SideLength**2,dtype =dtype) 
        xvalsOrdered = np.zeros(SideLength**2,dtype =dtype)
        yvalsOrdered = np.zeros(SideLength**2,dtype =dtype)
        resX = cl.array.to_device(queue, resx_np)
        resY = cl.array.to_device(queue, resy_np)
        for i in range(SideLength):
            yvalsOrdered[i*SideLength:i*SideLength+SideLength]=yvals[i]#every y val repeated side length times
            xvalsOrdered[i*SideLength:i*SideLength+SideLength]=xvals#whole xval arr repeated for each row
        Xs = cl.array.to_device(queue, xvalsOrdered)
        Ys = cl.array.to_device(queue, yvalsOrdered)  
        Hoppalongiterations = iterator(Consts, Xs, Ys, resY, resX,mapcl,N,SideLength,precompiled)
    else:
        xvals = np.array(x1,dtype=dtype)
        yvals = np.array(y1,dtype=dtype)
        resx_np = np.array(0,dtype =dtype)
        resy_np = np.array(0,dtype =dtype) 
        resX = cl.array.to_device(queue, resx_np)
        resY = cl.array.to_device(queue, resy_np)
        Xs = cl.array.to_device(queue, xvals)
        Ys = cl.array.to_device(queue, yvals)
        Hoppalongiterations = iterator(Consts, Xs, Ys, resY, resX,mapcl,N,SideLength,precompiled)
    return Hoppalongiterations

    




@njit
def points(Xs,Ys,XCount,YCount):
    Xbounds=np.linspace(np.min(Xs),np.max(Xs),XCount)#minimum Xvals to be assigned to pixel
    Ybounds=np.linspace(np.min(Ys),np.max(Ys),YCount)
    counts = np.zeros((XCount+2,YCount+2))
    for i in range(len(Xs)):
        counts[np.searchsorted(Xbounds,Xs[i],side="right"),np.searchsorted(Ybounds,Ys[i],side="right")]+=1
    return counts[1:-1,1:-1]


@njit(parallel=True)
def prepdataDiff(Hoppalongiterations,SideLength,Order=2):
    #hoppalong iterations is all of the points every hoppalong orbit visited so sidelength by side length by Niterations 
    resultrange=np.zeros((SideLength,SideLength))
    count=0
    progresscheck = int((SideLength**2)/10)
    for i in prange((SideLength**2)):
        count+=1
        if count%progresscheck == 0 and not i==0:#for progress indicator
            print((count/(SideLength**2))*100*100/8)
            print("%")
        rang=np.sum((np.ediff1d(Hoppalongiterations[:,0,i]))**Order+np.ediff1d(Hoppalongiterations[:,1,i])**Order)
        resultrange[i%SideLength,i//SideLength] = rang
    print(np.shape(resultrange))
    return resultrange

@njit(parallel=True)
def prepdataMaxMin(Hoppalongiterations,SideLength,width=200,height=200):
    #hoppalong iterations is all of the points every hoppalong orbit visited so sidelength by side length by Niterations 
    resultrange=np.zeros((SideLength,SideLength))
    count=0
    progresscheck = int((SideLength**2)/10)
    for i in prange((SideLength**2)):
        count+=1
        agg=points(Hoppalongiterations[:,0,i],Hoppalongiterations[:,1,i],width,height)#creates density map of locations of ith hoppalong orbit
        if count%progresscheck == 0 and not i==0:#for progress indicator
            print((count/(SideLength**2))*100*100/8)
            print("%")
            
        rangs = np.zeros(Hoppalongiterations.shape[0],dtype=np.float32)
        for j in range(Hoppalongiterations.shape[0]):
            rangs[j]=(np.max(agg[j,:])-np.min(agg[j,:]))#finds ranges of each jump
               
        rang=np.sum(rangs)
        resultrange[i%SideLength,i//SideLength] = rang#/dist
    return resultrange

def AttractorExplorer(x1,x2,y1,y2,N,mapclstr,SideLength,Res2 = 300,N2=40000,args = np.array([2.0,1.0,0.0],dtype =np.float32),iterator=iteratefast,dtype=np.float32):
    from matplotlib import gridspec
    if str(type(mapclstr[0]))=="<class 'pyopencl.elementwise.ElementwiseKernel'>":
        mapcl=mapclstr[0]
        ctx = mapclstr[1]
    elif dtype == np.float64:
        mapclstr[0] = mapclstr[0].replace("float","double")
        mapclstr[1] = mapclstr[1].replace("float","double")
        ctx = cl.create_some_context()
        mapcl = [ElementwiseKernel(ctx,*mapclstr,"mapcl"),cl.create_some_context(),cl.CommandQueue(ctx)]
    else:    
        ctx = cl.create_some_context()
        mapcl = [ElementwiseKernel(ctx,*mapclstr,"mapcl"),cl.create_some_context(),cl.CommandQueue(ctx)]
    
    plt.ion()
    extent = [x1,x2,y1,y2]
    Hoppalongiterations =iterateOpencl(*extent,N,mapcl,SideLength,args=args,precompiled=True)
    resultrange = prepdataMaxMin(Hoppalongiterations,SideLength)
    
    #Hoppalongiterations = points(Hoppalongiterations[:,0,0],Hoppalongiterations[:,1,0],1000,1000)
    fig=plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1],
         wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845) 
    #fig, ax = plt.subplots(2)
    ax=[plt.subplot(gs[0,0]),plt.subplot(gs[0,1])]
    #plt.tight_layout()
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
    ax[0].imshow(resultrange,extent=extent)
    Hoppalongorbit = iterateOpencl(0,0,0,0,N2,mapcl,1)
    orbit= points(Hoppalongorbit[:,0],Hoppalongorbit[:,1],500,500)#=  points(Hoppalongiterations[:,0,0],Hoppalongiterations[:,1,0],Res2,Res2)                
    ax[1].imshow(orbit)
    plt.draw()
    
    def onclick(event):
        
        print(event.key)
        Hoppalongorbit = iterateOpencl(event.xdata,0,event.ydata,0,N2,mapcl,1)
        orbit= points(Hoppalongorbit[:,0],Hoppalongorbit[:,1],500,500)#=  points(Hoppalongiterations[:,0,0],Hoppalongiterations[:,1,0],Res2,Res2)                
        
        ax[1].imshow(orbit)
        ax[1].xaxis.set_ticks_position('bottom')
        ax[1].yaxis.set_ticks_position('left')
        ax[1].spines['right'].set_color('none')
        ax[1].spines['top'].set_color('none')
        ax[1].spines['left'].set_position('center')
        ax[1].spines['bottom'].set_position('center')
        plt.draw()
        #print((event.xdata,event.ydata))
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True) 
    
    
    
if __name__ == '__main__':
    #time = timeit.default_timer()
    SideLength = 1000
    Consts = np.array([2,1,0],dtype =np.float32)
    #extent=np.array([(9 - np.sqrt(17))/8,10,(7+np.sqrt(17))/8,10])
    #centreofringsapprox=(1.4490920335076216, -0.4735961188883253)
    #print(prepAttractorCl(hoppalongx,hoppalongy))
    """centresoffirstlargerings= [(-0.2705627705627691, 5.974025974025977),
                               (4.761904761904766, 2.8354978354978364),
                               (7.846320346320351, -2.0887445887445875),
                               (4.707792207792213, -6.255411255411255),
                               (-0.2705627705627691, -6.255411255411255),
                               (-4.491341991341988, -2.1969696969696955),
                               (-4.491341991341988, 2.7272727272727284)]#found manually by clicking points on graph, not to be trusted 
    
    x,y=zip(*centresoffirstlargerings)
    extent=np.array([-10,10,-10,10])*5
    [x1,x2,y1,y2]=extent
    
    centre= np.mean(x),np.mean(y)
    #Hoppalongiterations =iterateOpencl(0,0,0,0,10000,mapclstr,1)
    #Hoppalongiterations = points(Hoppalongiterations[:,0],Hoppalongiterations[:,1],500,500)
    fig, ax = plt.subplots()
    #ax.scatter(*centre)
    #ax.scatter(extent[0:2], extent[2:])
    #ax.scatter(x,y)
    #ax.scatter(0, 0)
    
    time=timeit.default_timer()
    Hoppalongiterations = iterateOpencl(*extent,500,mapclstr,SideLength)
    print(timeit.default_timer()-time)
    #orbit= points(Hoppalongiterations[:,0],Hoppalongiterations[:,1],500,500)
    
    time=timeit.default_timer()
    resultrange = prepdataMaxMin(Hoppalongiterations,SideLength,width=300,height=300)
    print(timeit.default_timer()-time)
    
    #ax.imshow(orbit)
    plt.ion()
    ax.imshow(resultrange,extent=extent)
    #ax.scatter((9 - np.sqrt(17))/8,(7+np.sqrt(17))/8,marker="x",color="red")
    
    
    def onclick(event,pointofinterest=centre):
        
        print(event.key)
        clickedpoint=(event.xdata,event.ydata)
        dist=np.linalg.norm([clickedpoint,pointofinterest])
        print(f"clicked: {clickedpoint}")
        print(f"dist from POI is: {dist}")
        dist=np.linalg.norm([clickedpoint,(0,0)])
        print(f"dist from 0 is: {dist}")
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)
    face_id=0
    import os"""
    #while os.path.exists(r"C:\Users\kieran\Documents\Coding projects\Fractal-Viewer\figs\Hoppalong_maps\autosave"+str(face_id)+".svg"):
        #face_id+=1

    #f = r"C:\Users\kieran\Documents\Coding projects\Fractal-Viewer\figs\Hoppalong_maps\autosave"+str(face_id)+".svg"
    #print("saving to "+f)  
    #fig.savefig(f, format='svg', dpi=1200,bbox_inches='tight')
    

    





















