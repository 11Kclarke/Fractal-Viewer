
import numpy as np
import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel
import os
import matplotlib.pyplot as plt
from numba import jit,prange,njit
import pyopencl.array
import timeit
#//__f(X[i+ittlim*k],Y[i+ittlim*k],k1,k2,k3,k4,k5)__;
#//__g(X[i+ittlim*k],Y[i+ittlim*k],k1,k2,k3,k4,k5)__;
os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
hoppalongx = "k1-X[i]"
hoppalongy = "Y[i]-sign(X[i])*( sqrt(fabs(k2*X[i]-k3)))"
def prepAttractorCl(f,g,dtype=float):
    from PyToPyOpenCL import subsfunction
    mapclstr=["int ittlim,float k1,float k2, float k3,float k4,float k5,float *Y,float *X","""
        //i indexes row
        //k indexes column i,k-1 is value bellow, k=0 innitial
        
        
        for (int k=1; k<ittlim-1; k++){
            X[i*ittlim+k]=__f(X[i*ittlim+k-1],Y[i*ittlim+k-1],k1,k2,k3,k4,k5,X[i*ittlim],Y[i*ittlim])__;
            Y[i*ittlim+k]=__g(X[i*ittlim+k-1],Y[i*ittlim+k-1],k1,k2,k3,k4,k5,X[i*ittlim+k],Y[i*ittlim])__;
            
        }
        
        
        """]
    f=f.replace("y","Y[i]")
    f=f.replace("x","X[i]")
    g=g.replace("y","Y[i]")
    g=g.replace("x","X[i]")
    f=("float X[i],float Y[i],float k1,float k2, float k3,float k4,float k5,float X0, float Y0".split(","),f)
    g=("float X[i],float Y[i],float k1,float k2, float k3,float k4,float k5,float XN, float Y0".split(","),g)
    mapclstr[1]=subsfunction(f,mapclstr[1],"f",RemoveSemiColon=False)
    mapclstr[1]=subsfunction(g,mapclstr[1],"g",RemoveSemiColon=False)
    
    if dtype == np.float64:
            mapclstr[1] = mapclstr[1].replace("float","double")
            mapclstr[1] = mapclstr[1].replace("float","double")
            mapclstr[0] = mapclstr[0].replace("float","double")
            mapclstr[0] = mapclstr[0].replace("float","double")
    """for i in mapclstr[1].split("\n"):
        print(i)"""
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mapcl = ElementwiseKernel(ctx,*mapclstr,"mapcl")
    return mapcl,queue
            


def iterateOpencl(x1,x2,y1,y2,IttLim,mapclstr,SideLength,Consts = np.array([2.0,1.0,0.0,0.0,0.0],dtype =np.float32),dtype=np.float32):   
    if str(type(mapclstr[0]))=="<class 'pyopencl.elementwise.ElementwiseKernel'>":
        mapcl=mapclstr[0]
        queue = mapclstr[1]
        precompiled=True
    else:
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mapcl = ElementwiseKernel(ctx,*mapclstr,"mapcl")
        
    if SideLength>1:
        Xs,Ys=PrepAttractorData(x1,x2,y1,y2,SideLength,IttLim,dtype=dtype)
        Xs=cl.array.to_device(queue, Xs)
        Ys=cl.array.to_device(queue, Ys)
        Xs.size=SideLength*SideLength
        Ys.size=SideLength*SideLength
        mapcl(int(IttLim),*Consts,Ys,Xs)
        Xs=Xs.get().T
        Ys=Ys.get().T
        Hoppalongiterations=np.stack((Xs,Ys),axis=1)
    else:#for plotting the orbit
        Xs=np.zeros(IttLim)
        Ys=np.zeros(IttLim)
        Xs[0]=x1
        Ys[0]=y1
        Xs=cl.array.to_device(queue, Xs)
        Ys=cl.array.to_device(queue, Ys)
        Xs.size=SideLength*SideLength
        Ys.size=SideLength*SideLength
        mapcl(int(IttLim),*Consts,Ys,Xs)
        Xs=Xs.get().T
        Ys=Ys.get().T
        Hoppalongiterations=np.array([Xs,Ys])
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
    for i in prange((SideLength**2)):
        resultrange[i%SideLength,i//SideLength] = np.sum((np.ediff1d(Hoppalongiterations[:,0,i]))**Order+np.ediff1d(Hoppalongiterations[:,1,i])**Order)
    return resultrange



@njit(parallel=True)
def prepdataMaxMin(Hoppalongiterations,SideLength,width=200,height=200):
    #hoppalong iterations is all of the points every hoppalong orbit visited so sidelength by side length by Niterations 
    print("preping data")
    resultrange=np.zeros((SideLength,SideLength))
    for i in prange((SideLength**2)):
        agg=points(Hoppalongiterations[:,0,i],Hoppalongiterations[:,1,i],width,height)#creates density map of locations of ith hoppalong orbit   
        rangs = np.zeros(agg.shape[0],dtype=np.float32)
        for j in range(agg.shape[0]):
            rangs[j]=(np.max(agg[j,:])-np.min(agg[j,:]))#finds ranges of each jump
             
        rang=np.sum(rangs)
        resultrange[i%SideLength,i//SideLength] = rang#/dist
    #print("fin preping data")
    #print(np.shape(resultrange))
    return resultrange




def AttractorExplorer(x1,x2,y1,y2,N,f,g,SideLength,Res2 = 300,N2=40000,args = np.array([2.0,1.0,0.0,0.0,0.0],dtype =np.float64),dtype=np.float64,cmaporbit="jet",Aggfunc=prepdataMaxMin,orbitstart=(0,0)):
    from matplotlib import gridspec
    print((f,g))
    mapcl=prepAttractorCl(f,g,dtype=dtype)
    print(mapcl)
    plt.ion()
    extent = [x1,x2,y1,y2]
    Hoppalongiterations =iterateOpencl(*extent,N,mapcl,SideLength,Consts=args)
    resultrange = Aggfunc(Hoppalongiterations,SideLength)
    
    #Hoppalongiterations = points(Hoppalongiterations[:,0,0],Hoppalongiterations[:,1,0],1000,1000)
    fig=plt.figure()
    fig.set_size_inches(12,6)
    fig.set_label("Attractor Explorer")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1],
         wspace=0.2, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)
     
    #fig, ax = plt.subplots(2)
    ax=[plt.subplot(gs[0,0]),plt.subplot(gs[0,1])]
    
    #plt.tight_layout()
    ax[0].set_title("Function of Whole Domain/Heuristic for size of pattern")
    ax[1].set_title("Orbit of input Function X0,Y0 = "+str((orbitstart)))
    
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
    
    
    Orbit = iterateOpencl(orbitstart[0],0,orbitstart[1],0,N2,mapcl,1,Consts=args)
    Orbit = points(Orbit[0,:],Orbit[1,:],Res2,Res2)
    Orbit=  np.log2(Orbit,np.zeros_like(Orbit), where=(Orbit!=0))#=  points(Hoppalongiterations[:,0,0],Hoppalongiterations[:,1,0],Res2,Res2)                
    ax[1].imshow(Orbit,cmap=cmaporbit)
    plt.draw()
    
    def onclick(event):
        if event.key == "control":
            ax[1].set_title("Orbit of input Function X0,Y0 = "+str((round(event.xdata,4),round(event.ydata,4))))
            Orbit = iterateOpencl(event.xdata,0,event.ydata,0,N2,mapcl,1,Consts=args)
            Orbit = points(Orbit[0,:],Orbit[1,:],Res2,Res2)
            Orbit=  np.log2(Orbit,np.zeros_like(Orbit), where=(Orbit!=0))#=  points(Hoppalongiterations[:,0,0],Hoppalongiterations[:,1,0],Res2,Res2)                
            ax[1].imshow(Orbit,cmap=cmaporbit)
            ax[1].xaxis.set_ticks_position('bottom')
            ax[1].yaxis.set_ticks_position('left')
            ax[1].spines['right'].set_color('none')
            ax[1].spines['top'].set_color('none')
            ax[1].spines['left'].set_position('center')
            ax[1].spines['bottom'].set_position('center')
            plt.draw()
        else:
            print("Control click to generate new orbit originating from mouse coords")
    def onpress(event):
        if event.key == "m":
            extent=[*ax[0].get_xlim(),*ax[0].get_ylim()]
            print("regening map with borders")
            print(extent)
            Hoppalongiterations =iterateOpencl(*extent,N,mapcl,SideLength,Consts=args)
            Zoomedmap = Aggfunc(Hoppalongiterations,SideLength)
            ax[0].imshow(Zoomedmap,extent=extent)
            plt.draw()
        #print((event.xdata,event.ydata))
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid=fig.canvas.mpl_connect('key_press_event', onpress)
    plt.show(block=True) 

@njit(parallel=True)    
def PrepAttractorData(x1,x2,y1,y2,SideLength,IttLim,dtype):
    xvals = np.linspace(x1,x2,SideLength)
    yvals = np.linspace(y1,y2,SideLength)#creates start vals    
    xvalsOrdered = np.zeros(SideLength**2)#empty containers for start vals ordered
    yvalsOrdered = np.zeros(SideLength**2)
    for i in prange(SideLength):
        yvalsOrdered[i*SideLength:i*SideLength+SideLength]=yvals[i]#every y val repeated side length times
        xvalsOrdered[i*SideLength:i*SideLength+SideLength]=xvals#whole xval arr repeated for each row
    Xs=np.zeros((SideLength**2,IttLim))
    Ys=np.zeros((SideLength**2,IttLim))#includes space for itterated values
    Xs[:,0]=xvalsOrdered.astype(dtype)
    Ys[:,0]=yvalsOrdered.astype(dtype)
    return Xs,Ys
if __name__ == '__main__':
    
    SideLength = 500
    IttLim=50
    dtype=np.float64
    Consts = np.array([2,1,0],dtype =dtype)
    x1,x2,y1,y2=-2,2,-2,2
    extent=[x1,x2,y1,y2]
    hoppalongx = "k1- X[i]"
    hoppalongy = "Y[i]-sign(X[i])*( sqrt(fabs(k2*X[i]-k3)))"
    dejongx="sin(k1*Y[i])+cos(X[i]*k2)"
    dejongy="sin(k3*X[i])+cos(Y[i]*k4)"
    #info and start vals for https://www.researchgate.net/publication/263469351_ABOUT_WHAT_IS_CALLED_GUMOWSKI-MIRA_MAP
    Gumowski_MiraF = "(k2*x + (2*(1-k2)*x*x/(1.0 + x*x)))"
    Gumowski_Mirax="k1*y+k3*y*(1-k3*y*y)+"+Gumowski_MiraF
    Gumowski_Miray="-x+"+Gumowski_MiraF.replace("x","XN") #(k2 * XN + (2*(1 - k2)*XN*XN/(1.0 + XN*XN)))"
    AttractorExplorer(*extent,IttLim,Gumowski_Mirax,Gumowski_Miray,SideLength,Res2=500,N2=int(1e6),args=[1,np.cos(4*np.pi/5)+0.008,  0.1, 0, 0],orbitstart=(0, 0),Aggfunc=prepdataDiff)
    
    """
   
    time = timeit.default_timer()
    Xs,Ys=PrepAttractorData(*extent,SideLength,IttLim,dtype=dtype)
    print(timeit.default_timer()-time)
    #Xs[:,0]=np.array([1,2]).astype(dtype)
    #Ys[:,0]=np.array([1,2]).astype(dtype)
    
    Xs=cl.array.to_device(queue, Xs)
    Ys=cl.array.to_device(queue, Ys)
    
    Xs.size=SideLength*SideLength
    Ys.size=SideLength*SideLength
    mapcl(int(IttLim),2.0,1.0,0.0,0.0,0.0,Ys,Xs)
    Xs=Xs.get().T
    Ys=Ys.get().T
    print(timeit.default_timer()-time)
    Hoppalongiterations=np.stack((Xs,Ys),axis=1)
   """
    mapclstr=["float k1,float k2, float k3, float *Xs,float *Ys,float *resY,float *resX","""
    resY[i] =k1- Xs[i];
    resX[i]= Ys[i]-sign(Xs[i])*( sqrt(fabs(k2*Xs[i]-k3)));
    Xs[i]=resX[i];
    Ys[i]=resY[i];
    """]
    """
    #Hoppalongiterations = iterateOpencl(*extent,N,mapclstr,SideLength)
    print(timeit.default_timer()-time)
   
    
    resultrange = prepdataMaxMin(Hoppalongiterations,SideLength)
    print(timeit.default_timer()-time)
    plt.imshow(resultrange,extent=extent)
    
    plt.show()

"""


















