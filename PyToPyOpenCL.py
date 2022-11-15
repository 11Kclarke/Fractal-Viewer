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






#must obey order of operations
operations1arg=["-"]
operations2arg = ["^","/","*","+"]
replacements2arg = ["dtype_pow(base,arg)","dtype_divide(base,arg)","dtype_mul(base,arg)","dtype_add(base,arg)"]
replacements1arg = ["dtype_neg(arg)"]
outputprogram=""
outputatoms = []


def GetArg(arg,r=False):
    openbracet="("
    closebracet=")"
    if r:
        openbracet=")"
        closebracet="("
        arg=arg[::-1]
    opening=0
    closing=0
    for i,val in enumerate(arg):
        """print("val: "+val)
        print(arg[:i])
        print(not(arg[:i].isspace()))
        print(val.isspace())"""
        if val==closebracet:closing+=1
        if val==openbracet:opening+=1
        if opening < closing:# if opening < closing must have started in bracket
            #if r: print((arg,i,closebracet,openbracet))
            if not "," in arg[:i]:
                arg=arg[:i] 
            else: 
                arg=arg[:i].split(",",1)[0] 
            
            break
        if val.isspace() and not(arg[:i].isspace()) and len(arg[:i])>0:
            arg=arg[:i]
            break
    if r: arg=arg[::-1]
    return arg


def apply1ArgOps(operation,replacement,opp):
    arg=GetArg(operation[1:]).strip()
    regexp = re.compile(r"^(-?\d+)(\.\d*)?$")
    #regex to check for number currently rejects .123 but 0.123 is fine
    if bool(regexp.search(arg)):
        return opp+arg,arg
    else:
        return replacement.replace("arg",arg),arg
        

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

def dtyper(operation):#adds the r in the right place to use correct function call
    #funcname(real,comp) -->> rfuncname(real,comp)
    #funcname(come,real) -->> funcnamer(come,real)
    #funcname(comp,comp) -->> funcname(comp,comp)
    lhs=operation.find("_")
    rhs=operation.find("(")
    
    regexp = re.compile(r"^(-?\d+)(\.\d*)?$")#regex to check for number currently rejects .123 but 0.123 is fine 
    centralcommaindex=0
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
    #print(f"split into {args}")        
    assert not (regexp.search(args[0]) and regexp.search(args[1]))#no variable should have been simplified
    #regex to check for number currently rejects .123 but 0.123 is fine
    if bool(regexp.search(args[0])):
        operation = operation[:lhs+1]+"r"+operation[lhs+1:]
    if bool(regexp.search(args[1])):
        operation = operation[:rhs]+"r"+operation[rhs:]   
    return operation

def translate(fl,operations=operations2arg,replacements2arg=replacements2arg,outputarg = "",dtype="cdouble_",returnsig=True):
    if isinstance(fl,str):
        fl=sp.lamdify(fl)
    fsorig=inspect.getsource(fl)
    fsorig=fsorig.replace("**","^")#a special case poor solution, to deal with difficulty seperating * from **
    fsorig=fsorig.replace("- ","+-")#should do this for all 1arg funcs, however many like sin and cos or log would already have a + or - infront
    sig=makesig(fsorig.split("\n",1)[0],dtype)
    workingfs  = fsorig.split("\n",1)[1]
    workingfs = workingfs.replace("return"," ")
    
    for j,opp in enumerate(operations1arg):
        start=0
        c=workingfs.count(opp)#to avoid confusion editing variable while iterating
        workingfscopy=workingfs
        for i in range(c):
            index=workingfscopy[start:].find(opp)
            replacement,arg=apply1ArgOps(workingfscopy[index+start:],replacements1arg[j],opp)#if arg is numeric this func will return its input
            #print(f"replaced {opp+arg}, with {replacement}")
            workingfs=workingfs.replace(opp+arg,replacement).replace("dtype_",dtype)
            start=index+1
    if "*1j" in workingfs:#any complex constant will contain *1j eg 21*1j
        Coef=GetArg(workingfs[:workingfs.find("*1j")],r=True)
        genimag ="(dtype_t){0,arg}"#this is general for of complex constant for C complex libary being used
        #Took Fucking ages to work out how to do this apparently its called a compound literal whatever the fuck that means
        substitution=genimag.replace("arg",Coef).replace("dtype_",dtype)
        workingfs=workingfs.replace(Coef+"*1j",substitution)
        #print("replaced "+Coef+"*1j"+", with"+substitution)#for some reason f string wouldnt do this 
    c=0
    while c<len(operations):
        if operations[c] in workingfs:
            base,arg = workingfs.split(operations[c],1)
            base,arg = getBaseAndArgs(base,arg)
            replacement=replacements2arg[c].replace("dtype_",dtype).replace("base",base.strip()).replace("arg",arg.strip())
            replacement=dtyper(replacement)
            workingfs=workingfs.replace(base+operations[c]+arg,replacement)
        
        else:
            c+=1
    if returnsig: 
        return sig, workingfs 
    else:
        return workingfs
    
"""TO DO:
Fix bug with brackets inside of array index, causing inbetweenbrackets to be wrong
make matching of variables in function def to call sig more intuitive"""
def subsfunction(f,code,name,RemoveSemiColon=True):
    print("\n\n\n\n\n\nin subsfunc ")
    
    
    originalargs=[]

    sig,func=f
    #print(f)
    legalendchars=["=","-","/","*",",",")"]#characters allowed directly after variable
    #if there were a key word whose name contained a var name it wont be replaced unless next char is in list
    
    for i in legalendchars:#Adds spaces after end chars so variables can be identified 
                func=func.replace(i," "+i+" ")
                #code=code.replace(i," "+i+" ")
    
    for i in sig:#gets all variable names required for function being subbed
        originalargs.append(i.split(" ")[-1]+" ")
        
        #print((i.split(" ")[1]))
    #originalargs=sig[sig.find("(")+1:-2].split(",")  
    #print(originalargs)
    #print()  
    #print("code split")
    codesplit = code.split("__")
    #print(codesplit)
    
    for split in codesplit:
        if split.split("(")[0] == name:#split here will be the function name and its input eg "f(x)"
            print(split)
            strinbetweenbrackets=split.split("(")[1].split(")")[0]
            inbetweenbrackets=strinbetweenbrackets.split(",")
            print(inbetweenbrackets)
            #split 1 gets after first bracket, split to gets rid of after bracket, then forms list of each arg
            #inbetweenbrackets is equal to the list of argument names in function locations 
            #eg subbing somthing into __f(a,b)__ inbetween brackets would be ["a","b"]
            # strinbetweenbrackets = "a,b"
           
            assert len(originalargs)==len(inbetweenbrackets)
            
            for i in zip(originalargs,inbetweenbrackets):#replaces args in function def with arg in location
                #print("replacing ")
                #print(i[0].strip(),i[1].strip())
                func=func.replace(i[0].strip()+" ",i[1].strip())#the +" " is critical
                #it prevents variable names coincidentally appearing as substrings in other things being replaced
                #for example when c is a variable being replaced with Const c_double goes to Const_double
            if RemoveSemiColon:
                func=func.replace(";","")
             
            code=code.replace(f"__{split}__",func)
    print(code)
    print("leaving subsfunc \n\n\n\n\n\n")      
    return code

if __name__ == '__main__':#for testing not really intedned to be ran
    StabilityFractalNoCycle="""
    int Counter = 0;
    dtype_t Const=X[i];
    while (dtype_abs(X[i])>DivLim && Counter<N) 
    {
        Counter+=1;
        X[i]=__f(X[i],Const)__;
        Y[i]=__g(X[i],Const)__;
    }
   
    """
    
    x,c=sp.symbols("x,c")
    f=c*x**2
    fl=sp.lambdify((x,c),f)
    print(inspect.getsource(fl))
    flt=translate(fl)
    print(flt)
    mapclstr = subsfunction(flt,StabilityFractalNoCycle,"f")
    
    print(mapclstr)
    """   xbig,y=sp.symbols("xbig,y")
    f=xbig**2-xbig**3-1.0+xbig+21j

    fp = sp.diff(f)
    fl=sp.lambdify((xbig),f)

    fprimel=sp.lambdify(xbig,fp)
    flt=translate(fl)
    fprimelt = translate(fprimel)
    print(f)
    print(flt)
    print(fp)
    print(fprimelt)
     """
