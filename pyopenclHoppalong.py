
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


"""ith x and y should be unique, and map to ith res x res y"""

"""mapcl = ElementwiseKernel(ctx,
    "float k1,float k2, float k3, float *Xs,float *Ys,float *resY,float *resX",
    """
    
""" resY[i] =k1- Xs[i];
    resX[i]= Ys[i]-sign(Xs[i])*(sqrt(fabs(k2*Xs[i]-k3)));
    Xs[i]=resX[i];
    Ys[i]=resY[i];"""
""",
"mapcl")"""

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
def iterateOpencl(x1,x2,y1,y2,N,mapclstr,SideLength,args = np.array([2.0,1.0,0.0],dtype =np.float32),iterator=iteratefast,dtype=np.float32):
    precompiled=False
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
        
        print(SideLength)
        print("here")
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
        #for i in range(5):
            #mapclstr[1]+=mapclstr[1]
        #print("n less than 2")
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
def prepdata(Hoppalongiterations,SideLength,width=200,height=200):
    resultrange=np.zeros((SideLength,SideLength))
    count=0
    for i in prange((SideLength**2)):
        count+=1
        agg=points(Hoppalongiterations[:,0,i],Hoppalongiterations[:,1,i],width,height)
        if count%20000 == 0 and not i==0:
            print((count/SideLength**2)*100)
            print("%")
            
        rangs = np.zeros(width,dtype=np.float32)
        for j in range(agg.shape[0]):
            rangs[j]=(np.max(agg[j,:])-np.min(agg[j,:]))#+np.sum(np.max(agg[:,j])-np.min(agg[:,j]))
        resultrange[i%SideLength,i//SideLength] = np.sum(rangs)#np.sum(np.ptp(agg,axis=1))+np.sum(np.ptp(agg,axis=0))
    return resultrange

def AttractorExplorer(x1,x2,y1,y2,N,mapclstr,SideLength,Res2 = 400,N2=50000,args = np.array([2.0,1.0,0.0],dtype =np.float32),iterator=iteratefast,dtype=np.float32):
    from matplotlib import gridspec
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
        mapcl = [ElementwiseKernel(ctx,*mapclstr,"mapcl"),cl.create_some_context(),cl.CommandQueue(ctx)]
    else:    
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        
        mapcl = [ElementwiseKernel(ctx,*mapclstr,"mapcl"),cl.create_some_context(),cl.CommandQueue(ctx)]
    
    plt.ion()
    extent = [x1,x2,y1,y2]
    Hoppalongiterations =iterateOpencl(*extent,N,mapcl,SideLength)
    resultrange = prepdata(Hoppalongiterations,SideLength)
    
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
    time = timeit.default_timer()
    SideLength = 500
    Consts = np.array([2,1,0],dtype =np.float32)
    extent=np.array([-10,10,-10,10])
    [x1,x2,y1,y2]=extent
    mapclstr=["float k1,float k2, float k3, float *Xs,float *Ys,float *resY,float *resX","""
    resY[i] =k1- Xs[i];
    resX[i]= Ys[i]-sign(Xs[i])*(sqrt(fabs(k2*Xs[i]-k3)));
    Xs[i]=resX[i];
    Ys[i]=resY[i];
    """]
    
    #Hoppalongiterations =iterateOpencl(0,0,0,0,10000,mapclstr,1)
    #Hoppalongiterations = points(Hoppalongiterations[:,0],Hoppalongiterations[:,1],500,500)
    #Hoppalongorbit = iterateOpencl(0,0,0,0,10000,mapclstr,1)
    #rbit= points(Hoppalongiterations[:,0],Hoppalongiterations[:,1],500,500)
    #fig, ax = plt.subplots()
    #resultrange = prepdata(Hoppalongiterations,SideLength)
    #print(time-timeit.default_timer())
    #ax.imshow(Hoppalongiterations)
    #ax.imshow(resultrange,extent=extent)
    #plt.show()
    
    AttractorExplorer(x1,x2,y1,y2,50,mapclstr,SideLength)





















#need to have each y proccessed with each x not just yi with xi
""""ymap = ElementwiseKernel(ctx,
    "float k1, float *res_x,float *res_y",
    """"""res_y[i] =k1- res_x[j]"""""",
    "ymap")

xmap = ElementwiseKernel(ctx,
    "float k2, float k3, float *res_x,float *res_y",
    """""",
    "xmap")"""