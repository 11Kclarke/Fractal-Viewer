def genfract(function,boundLength,centre,npoints=200, maxdepth=100,complex=True,recurse=simprecurse):
    dx=boundLength/npoints
    xstartbounds = (centre[0]-boundLength/2,centre[0]+boundLength/2)
    ystartbounds = (centre[1]-boundLength/2,centre[1]+boundLength/2)
    print("herhehrehe")
    print(ystartbounds)
    return genfractfromvertex(function,xstartbounds[0],xstartbounds[1],ystartbounds[0],ystartbounds[1])

def Secondordernewtonsmethod(f,fprime,fsecond,x0,tol=1e-6,maxdepth=100):#written by autopilot dunno quite how it works
    x=x0
    for i in range(maxdepth):
        x=x-f(x)/fprime(x)-fsecond(x)/fprime(x)
        if abs(f(x))<tol:
            return x
    return x

def absdivergencedetector(ys,val=2):
 if ys[-1]>val:
     return True

def fractionRE(Zn,a):
    b=a[1]
    a=a[0]
    return 1/(a+b*Zn**2)

def fraction(Zn,a):
    b=a.imag
    a=a.real
    return 1/(a+b*Zn**2)

def Gradientdivergencedetector(ys,threshhold=3):
    diffs=[0]*(len(ys)-1)
    
    for i in range(len(ys)-1):
        #print(i)
        diffs[i]=ys[i]-ys[i+1]
    #print(ys)
    #print(diffs)
    for i in range(len(diffs)-1):
        if abs(diffs[i+1])>abs(threshhold*diffs[i]):
            return True #if difference greater then threshold ammounts of last diff, has diverged.
    return False