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
import sympy as sp
import inspect
import re


optable = {'+': lambda x, y: x + y,
      '-': lambda x, y: x - y,
      '/': lambda x, y: x / y,
      '*': lambda x, y: x * y,}



#must obey order of operations
operations2arg = ["^","/","*","+"]
#operations2arg=operations2arg[::-1]
replacements2arg = ["(dtype_#pow#(base,arg))","(dtype_#divide#(base,arg))","(dtype_#mul#(base,arg))","(dtype_#add#(base,arg))"]
#replacements2arg=replacements2arg[::-1]
#e^(i*x)="dtype_rpow(2.71828182,dtype_mul((dtype_t){0,1},arg))"
#e^(-i*x)="dtype_rpow(2.71828182,dtype_mul((dtype_t){0,-1},arg))"
#(e*iz-e^-iz)/2i=dtype_divide((dtype_t){0,2},dtype_add(e^(i*x),dtype_neg(e^(-i*x)))) = sin(x)
#(e*iz-e^-iz)/2i=dtype_rdivide(2,dtype_add(e^(i*x),e^(-i*x))) = cos(x)
operations1arg=["-",
                "sqrt",
                "exp",
                "tan",
                "sin",
                "cos"]
#maybe sin has 1 or 2 few brackets at end
replacements1arg = ["(dtype_neg(arg))",
                    "(dtype_powr(arg,0.5))",
                    "(dtype_rpow(2.71828182,arg))",
                    "sin(arg)/cos(arg)",
                    "(dtype_mul((dtype_t){0.0,1.0},(dtype_add(dtype_rmul(0.5,dtype_rpow(2.71828182,(dtype_mul((dtype_t){0.0,1.0},arg)))),dtype_neg(dtype_rmul(0.5,dtype_rpow(2.71828182,(dtype_neg(dtype_mul((dtype_t){0.0,1.0},arg))))))))))",
                    "(dtype_add(dtype_rmul(0.5,dtype_rpow(2.71828182,(dtype_mul((dtype_t){0.0,1.0},arg)))),dtype_rmul(0.5,dtype_rpow(2.71828182,(dtype_neg(dtype_mul((dtype_t){0.0,1.0},arg)))))))"]#cos(x)


"""Recomended process for adding new functions:
Add name\ identifier to operations list
Create operation in sympy from supported operations
Translate with Translate, put general dtype back in print then add to replacment list
Somthing like this:

f=(sp.exp(1j*x)+sp.exp(-1j*x))*0.5
fl=sp.lambdify(x,f)
flt=translate(fl)
flt=flt[1].replace("cdouble_","dtype_")
print("\n\n Cos(z)=")
print(flt)
----------
Cos(z)=
dtype_add(dtype_rmul(0.5,dtype_rpow(2.71828182,(dtype_mul((dtype_t){0.0,1.0},arg)))),dtype_rmul(0.5,dtype_rpow(2.71828182,(dtype_neg(dtype_mul((dtype_t){0.0,1.0},arg))))))"""

def removeextrabrackets(operation):
    while operation[0]=="(" and operation[-1]==")":
        operation = operation[1:-1]
    return operation


def GetArg(arg,r=False,diagnosticmode=False):
    specialchars=[" ",","]+operations2arg
    openbracet="("
    closebracet=")"
    reverse=0
    if r:
        openbracet=")"
        closebracet="("
        arg=arg[::-1]
        reverse=+1
    
    #this if might be worst line in whole project
    if (arg.find(openbracet)>arg.find(closebracet) or ((not (openbracet in arg)) and closebracet in arg)) and (not "," in arg[:arg.find(closebracet)]):
        #print(arg)
        #print(arg[:arg.find(closebracet)])
        
        #print(closebracet)
        #print("," in arg[:arg.find(closebracet)])
        arg=arg[:arg.find(closebracet)]#if close bracket appears before open bracket operator must have been in bracket
        #this is prolly why i had bracket count being compared to -1 not 0 in prev iterations
        #print("leaving from sketchy if statment")
    else:
        bracketcount=0
        startedcounting=False
        for i,val in enumerate(arg):
            
            if val==closebracet:
                startedcounting=True
                bracketcount-=1#should make it that if bracket counts ever negative it returns
                #would replace if before loop in dealing with case that operator and both args are in same bracket
            if val==openbracet:
                startedcounting=True
                bracketcount+=1
            if diagnosticmode:
                if r:
                    print(i,val)
                    print(arg[::-1])
                    print(((len(arg)-i)*" ") + ("-"*i))
                    print(f"bracket counnt = {bracketcount}")
                else:
                    print(i,val)
                    print(arg)
                    print("-"*i)
                    print(f"bracket counnt = {bracketcount}")
            if bracketcount==0:
                if startedcounting:
                    #for i in 
                    arg=arg[:i+1] 
                    break
                if val==",":
                    arg=arg[:i] 
                    break
            if val.isspace() and not(arg[:i+reverse].isspace()) and len(arg[:i+reverse])>0:#this may need indenting
                print(arg)
                print(arg[:i+reverse])
                print("\nleaving get args from space\n")
                arg=arg[:i+reverse]
                break    
    if r: arg=arg[::-1]
    #arg=removeextrabrackets(arg)
    return arg


def apply1ArgOps(operation,replacement,opp):
    """Weirdly harder then 2 args opp. Brackets around arg in most 1 arg funcs
    are seen as part of argument. eg sin=opp (x) = arg. In case of - theres usually no 
    brackets. to avoid harmless but messy redundant brackets being placed everywhere 
    extranious brackets removed from arg before being subbed into replacement, as
    replacements have required internal brackets in place already."""
    print(operation)
    print("into get args")
    arg=GetArg(operation[operation.find(opp)+len(opp):],diagnosticmode=False)
    #arg+=")"
    if arg[0]==opp:arg=arg[1:]
    assert arg.count("(")==arg.count(")")
    """print(arg)
    print("from")
    print(operation[operation.find(opp)+len(opp):])
    print("in")
    print(operation)
    print("replacing")
    print(opp+arg)"""
    regexp = re.compile(r"^\(?(-? ?\d+)(\.\d*)?\)?$")
    #regex to check for number currently rejects .123 but 0.123 is fine
    argextraniousbracketsremoved=removeextrabrackets(arg)
    #print(argextraniousbracketsremoved)
    replacement= replacement.replace("arg",argextraniousbracketsremoved)
    #replacement= replacement.replace("arg",arg.strip(")").strip("("))
    #print(replacement)   
    if bool(regexp.search(arg)):
        return opp+arg,arg
    else:
        return replacement,arg#special case needs replacing
        

def getBaseAndArgs(base,arg):#assuming bidmas is followed thing infront/after operator up to first space not in position 0 will be arg/base
    #eg a^b+c == (a^b) + c ^ has a,b and + has (a^b),c
    #if the order of operations fails we may end up with
    #a^(b+c)
    base=GetArg(base,r=True)
    arg=GetArg(arg)
    
    return base,arg

def makesig(pysig,dtype):
    pysig=pysig[pysig.find("(")+1:-2].split(",")
    symbols=[]
    for i in range(len(pysig)):
        symbols.append(pysig[i].strip())
        pysig[i]=dtype+" "+pysig[i]
    return pysig

def dtyper(args,operation):#adds the r in the right place to use correct function call
    #funcname(real,comp) -->> rfuncname(real,comp)
    #funcname(come,real) -->> funcnamer(come,real)
    #funcname(comp,comp) -->> funcname(comp,comp)
    lhs=operation.find("#")
    rhs=operation.rfind("#")
    regexp = re.compile(r"^\(?(-? ?\d+)(\.\d*)?\)?$")#regex to check for number currently rejects .123 but 0.123 is fine 
    #also allows for brackets around number for example (-1)
    """centralcommaindex=0
    bracketcount=0
    lowestbracketcomma=9999
    #print(operation)   
    if operation.count(",")>1:
        for i,val in enumerate(operation[rhs+1:]):#finds comma surounded by least brackets 
            if val==")":bracketcount-=1
            if val=="(":bracketcount+=1
            if val=="," and bracketcount<lowestbracketcomma:
                #print(bracketcount)
                centralcommaindex=i
                lowestbracketcomma=bracketcount       
        args=(operation[rhs+1:operation.rfind(")")][:centralcommaindex],
            operation[rhs+1:operation.rfind(")")][centralcommaindex+1:])#split whats between first opening bracket and last closing bracket by comma in least brackets
    else:
        args=(operation[rhs+1:operation.find(")")].split(","))
    print(f"{operation} \n split into \n{args}")"""
    if len(args)==1:
       
       return operation
    #if regexp.search(args[0]) and regexp.search(args[1]):
        #return operation        
    assert not (regexp.search(args[0]) and regexp.search(args[1]))#no variable should have been simplified, problem likely in get args, or apply 2 arg opp
    #regex to check for number currently rejects .123 but 0.123 is fine
    if len(args)==1:
        #print(operation)
        return operation
    if bool(regexp.search(args[0])):
        print("base real")
        print(args[0])
        operation = operation[:lhs+1]+"r"+operation[lhs+1:]
    if bool(regexp.search(args[1])):
        print("arg real")
        print(args[1])
        operation = operation[:rhs]+"r"+operation[rhs:]
    operation=operation.replace("#","")   
    return operation




def translateRegion(workingfs,operations=operations2arg,replacements2arg=replacements2arg,outputarg = "",dtype="cdouble_",returnsig=True):
    regexp = re.compile(r"^\(?(-? ?\d+)(\.\d*)?\)?$")#regex to check for number currently rejects .123 but 0.123 is fine 
    #also allows for brackets around number for example (-1)
    for j,opp in enumerate(operations1arg):
        start=0
        c=workingfs.count(opp)#to avoid confusion editing variable while iterating
        #print(workingfs)
        #print(opp)
        assert workingfs.count("(")==workingfs.count(")")
        workingfscopy=workingfs
        for i in range(c):
            index=workingfscopy[start:].find(opp)
            replacement,arg=apply1ArgOps(workingfscopy[index+start:],replacements1arg[j],opp)#if arg is numeric this func will return its input
            
            #print(f"replaced {opp+arg}, with {replacement}")
            workingfs=workingfs.replace(opp+arg,replacement).replace("dtype_",dtype)
            
            start=index+1
    
    c=0
    realoperationcount=0
    while c<len(operations):
        assert workingfs.count("(")==workingfs.count(")")
        if  workingfs.count(operations[c])>realoperationcount:
            base,arg = workingfs.split(operations[c],1)
            #ive removed and retyped these prints to many times i wont take them out again
            print("\n\n\n")
            print(workingfs)
            print((workingfs.count("("),workingfs.count(")")))
            print("into getargs:")
            print(base)
            print(operations[c])
            print(arg)
            print((base.count("("),base.count(")")))    
            #base=GetArg(base,r=True,diagnosticmode=True)    
            base,arg = getBaseAndArgs(base,arg)
            print("extracted args:")
            print(base)
            print(arg)
            print((base.count("("),base.count(")")))
            #assert operations[c]!="+"
            if not(bool(regexp.search(arg)) and bool(regexp.search(base))):#only replace if atleast 1 arg is complex type
                replacement=dtyper((base,arg),replacements2arg[c])
                
                replacement=replacement.replace("dtype_",dtype).replace("base",base.strip()).replace("arg",arg.strip())
                #print(f"replacing {base+operations[c]+arg} with {replacement}")
                 
            else:
                realoperationcount+=1
                
                replacement=optable[operations[c]](base,arg)
                #print(f"Simiplifying {base} and {arg} into, {replacement} as as both real")
                #print(f"{workingfs.count(operations[c])} number of {operations[c]}")
                #print(realoperationcount)
            workingfs=workingfs.replace(base+operations[c]+arg,replacement)    
        else:
            c+=1
            realoperationcount=0
    
    return workingfs

def translate(f,extraPrecisionArgs=0,operations=operations2arg,replacements2arg=replacements2arg,outputarg = "",dtype="cdouble_",returnsig=True):
    x,c,y,a,b,d=sp.symbols("x,c,y,a,b,d")
    extraPrecisionArgsStorage=[]
    if extraPrecisionArgs>0:
        print(f)
        for i in range(extraPrecisionArgs):
            f=f.subs(x,"extrprecision"+str(i)+"+ x")
            f=f.subs(c,"extrprecision"+str(i)+"+ c")
            f=f.expand()
            extraPrecisionArgsStorage.append("extrprecision"+"["+str(i)+"]")
            
            print(extraPrecisionArgsStorage)
    print(f.atoms(sp.Symbol))
    symbolsinF=list(f.atoms(sp.Symbol))
    #if x in symbolsinF:
        #symbolsinF[0],symbolsinF[symbolsinF.index(x)]=symbolsinF[symbolsinF.index(x)],symbolsinF[0]   
    print(f)
    fl=sp.lambdify(symbolsinF,f)
    print(fl)
    
    fsorig=inspect.getsource(fl)
    fsorig=fsorig.replace("**","^")#a special case poor solution, to deal with difficulty seperating * from **
    fsorig=fsorig.replace("- ","+-")#should do this for all 1arg funcs, however many like sin and cos or log would already have a + or - infront
    
    sig=makesig(fsorig.split("\n",1)[0],dtype)
    #for i in extraPrecisionArgsStorage:
        #sig.append(","+dtype+" "+i)
    print(sig)
    workingfs  = fsorig.split("\n",1)[1]
    workingfs = workingfs.replace("return","").strip()
    workingfs=workingfs.replace("1.0*","")
    bracketcount=0
    bracketedregions = [[workingfs,0]]#entire func furthest outer function
    openingindexes = []
    for i,val in enumerate(workingfs):#splits into bracketed regions
        if val==")":
            bracketcount-=1 
            bracketedregions.append([workingfs[openingindexes.pop()+1:i],bracketcount+1])#appending (region,depth)        
        elif val=="(":
            bracketcount+=1
            openingindexes.append(i)
    
    #print(bracketcount)
    assert bracketcount == 0
    
    bracketedregions.sort(key=lambda a: a[1],reverse=True)
    #sorts so brackets deepest get translated first
    bracketedregions=[i[0] for i in bracketedregions]  
    
    for i in range(len(bracketedregions)):
        val=bracketedregions[i]#cant use enumerate as itterable is edited during loop
        #print(f"translating {val}")
        translated = translateRegion(val)
        workingfs=workingfs.replace(val,translated)
        assert workingfs.count("(")==workingfs.count(")")
        for j in range(len(bracketedregions[i+1:])):#applies change to other regions 
            bracketedregions[i+1+j]=bracketedregions[i+1+j].replace(val,translated)        
    """a better way to ensure getargs doesnt need to deal with complex
    statments would be write a function that checks if somthing contains 
    an operator still and call translate region on that before passing to get arg
    atm this copys each bracketed region, translates from inside to out.
    this seems to work but requires reduncdantly storing of regions and applying translation to each
    would prolly be quite slow in a bracket hell"""
    #print(workingfs)     
    if "1j" in workingfs:#any complex constant will contain *1j eg 21*1j
        #Coef=GetArg(workingfs[:workingfs.find("*1j")],r=True)
        genimag ="(dtype_t){0.0,1.0}"#this is general for of complex constant for C complex libary being used
        #Took Fucking ages to work out how to do this apparently its called a compound literal whatever the fuck that means
        substitution=genimag.replace("dtype_",dtype)
        workingfs=workingfs.replace("1j",substitution)
        #print("replaced "+Coef+"*1j"+", with"+substitution)#for some reason f string wouldnt do this     
    orderedsig=[]
    if extraPrecisionArgs>0:
      
        print("\n\n\n\n end of translate \n\n\n")
        sigstr=" ,".join(sig)
        for i in range(extraPrecisionArgs):
            workingfs=workingfs.replace("extrprecision"+str(i),"extrprecision"+"["+str(i)+"]")
            sigstr=sigstr.replace("extrprecision"+str(i),"extrprecision"+"["+str(i)+"]")
        sig=sigstr.split(",")      
    if returnsig: 
        return sig, workingfs 
    else:
        return workingfs
"""TO DO:
Fix bug with brackets inside of array index, causing inbetweenbrackets to be wrong
make matching of variables in function def to call sig more intuitive"""
def subsfunction(f,code,name,RemoveSemiColon=True,sig=None):
    autosig,func=f
    if sig==None:
        sig=autosig
    print("func in")
    print(f)
    print(sig)
    print("to this code")
    print(code)
    
    originalfuncargs=[]
    for i,val in enumerate(sig):#gets all variable names required for function being subbed
        if "[" in val:
            val=val[:val.find("[")]
        originalfuncargs.append(val.strip().split(" ")[-1]+" ")
    print(originalfuncargs)
    
    #print(f)
    legalendchars=["=","-","/","*",",",")"]#characters allowed directly after variable
    #if there were a key word whose name contained a var name it wont be replaced unless next char is in list
    
    for i in legalendchars:#Adds spaces after end chars so variables can be identified 
                func=func.replace(i," "+i+" ")
                #code=code.replace(i," "+i+" ")
   
    
   
    
    #print()  
    print("code split")
    codesplit = code.split("__")
    print(codesplit)
    
    for split in codesplit:
        if split.split("(")[0] == name:#split here will be the function name and its input eg "f(x)"
            
            strinbetweenbrackets=split.split("(")[1].split(")")[0]
            inbetweenbrackets=strinbetweenbrackets.split(",")
            
            #split 1 gets after first bracket, split to gets rid of after bracket, then forms list of each arg
            #inbetweenbrackets is equal to the list of argument names in function locations 
            #eg subbing somthing into __f(a,b)__ inbetween brackets would be ["a","b"]
            # strinbetweenbrackets = "a,b"
           
            #assert len(originalargs)==len(inbetweenbrackets)
            
            for i in zip(originalfuncargs,inbetweenbrackets):#replaces args in function def with arg in location
                if "[" in i[1]:
                    print(inbetweenbrackets)
                    func=func.replace(i[0].strip(),i[1].strip().strip("]").strip("["))
                    #i[1]=inbetweenbrackets
                else:    
                    print("replacing ")
                    print(i[0].strip(),i[1].strip())
                    func=func.replace(i[0].strip()+" ",i[1].strip())#the +" " is critical
                #it prevents variable names coincidentally appearing as substrings in other things being replaced
                #for example when c is a variable being replaced with Const c_double goes to Const_double
                # a better solution would find a instance of substring then somehow check adjacent to it
            if RemoveSemiColon:
                func=func.replace(";","")
             
            code=code.replace(f"__{split}__",func)
    return code

if __name__ == '__main__':#for testing not really intedned to be ran
    os.environ["PYOPENCL_CTX"]="0"
    mapclstr="""



    X[i]=__f(X[i])__



    """
    #print(GetArg("cdouble_neg(cdouble_rmul(1.0,1j",r=True))

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    A=np.linspace(0,np.pi*6,1000,dtype=np.complex128)#1val for easy testing
    x,extrprecision0,c=sp.symbols("x,extrprecision[0],c1")
    #f=(sp.exp(1j*x)+sp.exp(-1j*x))*0.5
    #f=sp.exp(x)
    #f=(sp.exp(1j*x)+sp.exp(-1j*x))*0.5#
    
    extraprecision=1
    f=x**2+c
   
    
    #fl=sp.lambdify(x,f)
    #print(f)
    #print(inspect.getsource(fl))
    flt=translate(f,extraPrecisionArgs=1)
   
        
    print(flt)
    
    """
    cosgenimage=flt[1].replace("cdouble_","dtype_").replace("x","arg")
    print("\n\nCos(arg)=\n")
    print(cosgenimage)
    print("\n\n\n")
    """
    
    #mapclstr = subsfunction(flt,mapclstr,"f")
    #A=np.arange(25).reshape((5,5))

    
    
    #res_g = cl.array.to_device(queue, A)


    #mapcl = ElementwiseKernel(ctx,"cdouble_t *X,cdouble_t c",mapclstr,"mapcl",preamble="#define PYOPENCL_DEFINE_CDOUBLE //#include <pyopencl-complex.h>  ")
    #mapcl(res_g,1)
    #print(res_g.get())
    #cos=res_g.get()
    #plt.plot(cos)
    #plt.show()
    """ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    A=np.linspace(0,np.pi*6,1000,dtype=np.complex128)#1val for easy testing
    
    f=(sp.exp(1j*x)-sp.exp(-1j*x))*0.5*1j
    fl=sp.lambdify(x,f)
    flt=translate(fl)
    singenimage=flt[1].replace("cdouble_","dtype_").replace("x","arg")
    print("\n\nSin(arg)=\n")
    print(singenimage)
    print("\n\n\n")
    print(inspect.getsource(fl))
    mapclstr="""



    #X[i]=__f(X[i])__



    """
    mapclstr = subsfunction(flt,mapclstr,"f")
    #A=np.arange(25).reshape((5,5))


    
    res_g = cl.array.to_device(queue, A)


    mapcl = ElementwiseKernel(ctx,"cdouble_t *X",mapclstr,"mapcl",preamble="#define PYOPENCL_DEFINE_CDOUBLE //#include <pyopencl-complex.h>  ")
    mapcl(res_g)
    #print(res_g.get())
    plt.plot(res_g.get())
    plt.show()"""
    
    
