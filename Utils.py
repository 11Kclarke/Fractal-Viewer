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

#@njit(parallel=True)
def createstartvals(x1,x2,y1,y2,SideLength,ExtraPrecisionVars=[]):#this essentially creates a flattened grid for using as input into pyopenclfuncs
    Xlength=int(np.ceil(abs((x1-x2)/(y1-y2)*SideLength)))
    Ylength=int(np.ceil((SideLength**2)/Xlength))
    print(abs((x2-x1)/Xlength))
    print(abs((y2-y1)/Ylength))
    print(Xlength)
    print(Ylength)
    print("pre Extraprecision")
    print((x1,x2,y1,y2))
    
    if len(ExtraPrecisionVars)>0 and (abs((x2-x1)/Xlength) <1e-3 or abs((y2-y1)/Ylength) <1e-3):
        for i,val in enumerate(ExtraPrecisionVars):
            if abs(val)==0:
                ExtraPrecisionVars[i]=complex(x1,y1)#bottom left corner
        #ExtraPrecisionVars.append(complex(x1,y1))
        print("\n\n\n\n\nAdding New Variable For extra Depth\n\n\n\n")
        print(x1<x2)
        print(y1<y2)
        print(ExtraPrecisionVars[i])
        x2=-(x2-x1)
        print(x2)
        x1=0
        print(x1)
        y2=(y2-y1)
        print(y2)
        y1=0
        print(y1)
        print("adjusted extent")
        print((x1,x2,y1,y2))
        print("With extra precision var included")
        print((x1+ExtraPrecisionVars[0].real,x2+ExtraPrecisionVars[0].real,y1+ExtraPrecisionVars[0].imag,y2+ExtraPrecisionVars[0].imag))
        #Ylength/Xlength must be conserved
        #x2-x1 ----> (x2-x1)/2--(x2-x1)/2=x2-x1
        #y2-y1 ----> (y2-y1)/2--(y2-y1)/2=y2-y1
        # Xlength and Ylength conserved even better than ratio conserved
        
        #an extraprecisionvar = first 14 sigfigs
        #Vals only needs to be able to store sigfigs after 14 most significant.
        #Another extraprecisionvar will need to be added after another 14 digits
        #This should work until 304 digits, at which point the 64 bit nums storing 
        #Vals cant get any smaller, an extra scaling variable will be needed taking 
        #the load of the exponent. If Vals --> 1e-304*Vals, Vals can be near 0.
        #Further 14 digits will require a scaling variable to be stored with its extraprecision var
        #every 14 digits add extraprecisionvar storing current 14 bits, after every 300 multiply Current 
        #14 bits by 1e-300
        
    
    Xs=np.linspace((-x1)*1j,(-x2)*1j,Xlength).astype(complex)
    #Xs=np.linspace(x1*1j,x2*1j,Xlength).astype(complex)
    Ys=np.linspace(y1,y2,Ylength).astype(complex)
    if abs(ExtraPrecisionVars)!=0:
        if not(all(Xs.imag>=0) and all(Ys.real>=0)):
            print(Xs)
            print(Ys)
            print(Xs.imag>=0)
            print(Ys.real>=0)
    Vals=np.zeros(Xlength*Ylength).astype(complex)
    """for i,x in enumerate(Xs):
        for j,y in enumerate(Ys):
            Vals[i,j]=x+y"""
            
            
    """mid=((x1+x2)/2,(y1+y2)/2)
    x1,y1=x1,y1-mid
    x2,y2=x2,y2-mid
    xstretch=1/x1-x2
    ystretch=1/(y1-y2)
    x1,x2=x1*xstretch,x2*xstretch
    y1,y2=y1*ystretch,y2*ystretch"""
    extent=np.array([x1,x2,y1,y2])
    
    for i in range(Xlength):
        Vals[i*Ylength:i*Ylength+Ylength]=Ys
        Vals[i*Ylength:i*Ylength+Ylength]+=Xs[i]#every y val repeated side length times
    #if len(ExtraPrecisionVars)>0:
    """if abs(ExtraPrecisionVars)!=0:
        assert all(Vals.real>=0)
        assert all(Vals.imag>=0)"""
    return Vals,(Xlength,Ylength),ExtraPrecisionVars,extent
    #else:
        #return Vals,(Xlength,Ylength)