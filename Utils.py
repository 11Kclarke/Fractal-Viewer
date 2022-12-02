from numba import njit,prange
import numpy as np

def createstartvalssimp(x1,x2,y1,y2,SideLength):#this essentially creates a flattened grid for using as input into pyopenclfuncs
    Xs=np.linspace(x1,x2,SideLength,dtype=np.complex128)
    Ys=np.linspace(y1*1j,y2*1j,SideLength)
    Vals=np.zeros(SideLength*SideLength,dtype=np.complex128)
    """for i,x in enumerate(Xs):
        for j,y in enumerate(Ys):
            Vals[(i+1)*(j+1)]=x+y"""
    for i in range(SideLength):
        Vals[i*SideLength:i*SideLength+SideLength]=Xs
        Vals[i*SideLength:i*SideLength+SideLength]+=Ys[i]#every y val repeated side length times"""
    return Vals#can be reshaped into square grid    
"""Takes complex to complex function, and complex value, 
and returns recursion count before value > lim"""
#@njit
def FindreferenceOrbit(function, maxdepth, X0,dx,dy,DivLim=2):
    
    dx=abs(dx)
    dy=abs(dy)
    X0in=X0
    bestX0=X0
    bestys=np.zeros(maxdepth,dtype=complex)
    longest=0
    moveCount=0
    SpiralLength=1
    ys=np.zeros(maxdepth,dtype=complex)
    #ys[0]=complex(0,0)
    print(X0)
    print(DivLim)
    print(maxdepth)
    print((dx,dy))
    i=1
    k=0
    while i<maxdepth and k<2*maxdepth:
        
        if abs(ys[i-1])>DivLim:#if diverges early
            #print("Diverged")
            k+=1
            if i > longest:
                #print(f"New best Orbit is {i} long")
                longest=i
                bestX0=X0
                bestys[:]=ys[:]
            #pick new x0    
            if moveCount<SpiralLength:    
                X0=complex(X0.real+dx,X0.imag)
                moveCount+=1
            elif moveCount<2*SpiralLength:
                X0=complex(X0.real,X0.imag+dy)
                moveCount+=1
            else:
                dx*=-1
                dy*=-1
                SpiralLength+=1
                moveCount=0
            #new X0 picked
            
            i=1
            ys=np.zeros(maxdepth,dtype=complex)
            ys[i]=function(0,X0)
        else:
            
            ys[i]=function(ys[i-1],X0)
            i+=1
    if k ==0 or i>= maxdepth:
        longest=i
        bestX0=X0
        bestys[:]=ys[:]     
           
    print("reference Orbit info:")
    print(bestX0)
    print(bestys[0])
    print(bestys[1])
    print(bestys)
    print(X0in)
    print(abs(bestX0-X0in))
    print(f"Diverges after {longest}")
    print(f"checked {k} startvals")
    print()
    """import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    Ys=np.imag(Xs)
    Xs=np.real(Xs)
    ax.scatter(Xs,Ys,s=0.01)
    for i in range(len(Xs)):
        ax.annotate(str(i),(Xs[i],Ys[i]))
    plt.show()    """
    return bestys

    
    

#@njit(parallel=True)
def CreateStartVals(x1,x2,y1,y2,SideLength,ReferenceOrbit=None,Func=None,maxdepth=None,DivLim=None):#this essentially creates a flattened grid for using as input into pyopenclfuncs
    #Xlength=int(np.ceil(abs((x1-x2)/(y1-y2)*SideLength)))
    #Ylength=int(np.ceil((SideLength**2)/Xlength))
    Xlength=SideLength
    Ylength=SideLength
    reldiffs=np.array([abs((x1-x2)/(x1*Xlength)),abs((x2-x1)/(x2*Xlength)),
              abs((y1-y2)/(y1*Ylength)),abs((y2-y1)/(y2*Ylength))])
    PixelsToClose=any(reldiffs<1e-16)#minimum relative difference of x1,x2
    print("relative diff extent")
    print(reldiffs) 
    if isinstance(ReferenceOrbit,np.ndarray) and PixelsToClose:
        print("finding new Reference Orbit")
        xref0=ReferenceOrbit[1].real
        yref0=ReferenceOrbit[1].imag
        x1+=xref0
        x2+=xref0
        y1+=yref0
        y2+=yref0              
    if PixelsToClose:# and not isinstance(ReferenceOrbit,np.ndarray):#make reference orbit
        print("\n\nentering perturbation mode\n\n")
        print(f"extent in : {(x1,x2,y1,y2)}")
        dx=abs(x1-x2)
        dy=abs(y1-y2)
        print(f"dy and dx (abs) {dy,dx}")
        #reference orbit X0 chosen to minimize dist to other points
        midx=(x1+x2)/2
        midy=(y1+y2)/2
        #Converts Vals to Delata 0 of each pixel
        ReferenceOrbit=FindreferenceOrbit(Func,maxdepth,complex(midx,midy),dx,dy,DivLim=DivLim)
        xref0=ReferenceOrbit[1].real
        yref0=ReferenceOrbit[1].imag
        
        print(f"reference orbit 0 {ReferenceOrbit[0]}")
        
        x1-=xref0
        x2-=xref0
        y1-=yref0
        y2-=yref0
        dx=abs(x1-x2)
        dy=abs(y1-y2)
        print(f"dy and dx (abs) out {dy,dx}")
        print(f"extent out : {(x1,x2,y1,y2)}\n\n")
       
        reldiffs=np.array([abs((x1-x2)/(x1*Xlength)),abs((x2-x1)/(x2*Xlength)),
              abs((y1-y2)/(y1*Ylength)),abs((y2-y1)/(y2*Ylength))])
        print("\nrelative diff extent out\n")
        print(reldiffs)      
    #Xs=np.linspace(-x1*1j,-x2*1j,Xlength).astype(np.complex128)
    #Ys=np.linspace(y1,y2,Ylength).astype(np.complex128)
    print(Xlength)
    print(Ylength)
    Xs=np.linspace(x1,x2,Xlength).astype(np.complex128)
    Ys=np.linspace(y1*1j,y2*1j,Ylength).astype(np.complex128)
    Vals=np.zeros(Xlength*Ylength).astype(np.complex128)
    
    """for i,x in enumerate(Xs):
        for j,y in enumerate(Ys):
            Vals[i,j]=x+y"""
    
    for i in prange(Xlength):
        Vals[i*Ylength:i*Ylength+Ylength]=Ys
        Vals[i*Ylength:i*Ylength+Ylength]+=Xs[i]#every y val repeated side length times
    return Vals,(Xlength,Ylength),ReferenceOrbit