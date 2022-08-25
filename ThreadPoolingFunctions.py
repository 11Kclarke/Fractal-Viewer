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

def genfractfromvertex(function,x1,x2,y1,y2,npoints=500, maxdepth=150,iscomplex=True,iterator=iterator):
    #print("Generating fractal from vertex")
    
    
    stabilities = np.ones((npoints,npoints))*maxdepth
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
                stabilities[xc,yc]=iterator(function,maxdepth,(x+y),D,#complex numbers must be combined
                divergencedetector=absdivergencedetectoroptimised,cycledetector=cycledetector)
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