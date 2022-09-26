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




def isvalidsymbol(line,symbols):#is only a valid symbol if its a number or a variable seen in signature
    if isinstance(line,list):
        line="".join(line) 
    if line in symbols or line.isnumeric():
        return True
    else:
        return False
operations = ["^"]
operationsOrig = ["base^arg"]
replacements = ["pow( base , arg )"]
outputprogram=""
outputatoms = []
def getargs(operation,atom,symbols):
    operationSplit=atom.split(operation)
    """for i, val in enumerate(operationSplit):#splits multiplications into 2 symbols for checking
        if "*" in val:
            val=val.split("*")
            operationSplit[i]=val[0]
            operationSplit+=val[1:]"""
    base="temp"
    arg="temp"
    if isvalidsymbol(operationSplit[0].strip(),symbols):
        base=operationSplit[0].strip()                
    else:
        for m in range(1,len(operationSplit[0])):
            if isvalidsymbol(operationSplit[m:],symbols):
                base=operationSplit[m:]
                break
            if base=="temp":
                print("cannot find base for operation: "+str(operation)+"in "+atom)
    if isvalidsymbol(operationSplit[1].strip(),symbols):
                arg=operationSplit[1].strip()
    else:
        for m in range(1,len(operationSplit[1])):
            if isvalidsymbol(operationSplit[0:m],symbols):
                base=operationSplit[0:m]
                break
            if arg=="temp":
                print("cannot find arg for operation: "+str(operation)+" in "+atom )                       
    return arg,base


"""This is pretty shit and also a big mess rn
To do somthing for dealing with multiplication that doesnt break powers, i think rn A**n * B**m would break
needs to split 2 atoms multiplied A*B --> A * B without A**B --> A* *B"""
def translate(fl,operations=operations,operationsOrig=operationsOrig,outputarg = ""):
    print(inspect.getsource(fl))
    fsorig=inspect.getsource(fl)
    fsorig=fsorig.replace("**","^")#a special case poor solution, to deal with difficulty seperating * from **
    fs=fsorig.splitlines()
    workingfs  = fsorig.split("\n",1)[1]
    
        
    pysig=fs[0]
    pysig=pysig[pysig.find("(")+1:-2].split(",")
    symbols=[]
    for i in range(len(pysig)):
        symbols.append(pysig[i].strip())
        pysig[i]=dtype+" "+pysig[i]
    if len(outputarg)>0:
        pysig.append(dtype+" "+outputarg)
        workingfs=workingfs.replace("return", outputarg+" = ")
    else:
        workingfs=workingfs.replace("return", "")
    pysig=", ".join(pysig)
    for i in range(1,len(fs)):
        fsSplit=re.split(" |\*",fs[i])#in general sp.lamdify splits different operations by spaces, the main exception is two atoms multiplied
        #print(fsSplit)
        outputatoms.append([])#each row is a list of all opencl representations, of every operation in line
        for k in range(len(operations)):#list of operations that need replacing
            for atom in fsSplit:#example atom n*var**m
                if operations[k] in atom:
                    print(atom)            
                    arg,base=getargs(operations[k],atom,symbols)
                    print(arg,base)
                    if not base  =="temp" and not arg=="temp":#should only be 1 operation per substring seperated by spaces, this fails with brackets
                        replacement=replacements[k].replace("base",base)
                        replacement=replacement.replace("arg",arg)
                        outputatoms[i-1].append(replacement)
                        tobereplaced=operationsOrig[k].replace("base",base)
                        tobereplaced=tobereplaced.replace("arg",arg)
                        symbols.append(replacement)
                        workingfs=workingfs.replace(tobereplaced,replacement)
                        #break
                    else:
                        print("\n\nFailed to find arg and base \n\n")
    workingfs=workingfs.replace("\n",";\n")
    return pysig,workingfs
     
    


def subsfunction(f,code,name):
    originalargs=[]
    sig,func=f
    sig=sig.split(',')
    for i in sig:#gets all variable names required for function being subbed
        originalargs.append(i.split(" ")[1])
    codesplit = code.split("__")
    for split in codesplit:
        if split.split("(")[0] == name:#split here will be the function name and its input eg "f(x)"
            inbetweenbrackets=split.split("(")[1].split(")")[0].split(",")
            #split 1 gets after first bracket, split to gets rid of after bracket, then forms list of each arg
            assert len(originalargs)==len(inbetweenbrackets)
            for i in zip(originalargs,inbetweenbrackets):#replaces args in function def with arg in location
                func=func.replace(*i)
            code=code.replace(f"__{split}__",func)
            
        
        
        
    return code
#printf("%f",fval);
NewtonsMethod="""
int C = 0;
float fval=1000;
while (fval>precision && C<N) 
{
  fval=__f(X[i])__
  
  X[i] =X[i] - fval/__fprime(X[i])__ 
  
  C+=1;
}"""

dtype="float"
funcdefmarker="def _lambdifygenerated"
xbig,y=sp.symbols("xbig,y")
f=xbig**2+xbig**3-1

fp = sp.diff(f)
fl=sp.lambdify((xbig),f)

fprimel=sp.lambdify(xbig,fp)
flt=translate(fl)
fprimelt = translate(fprimel)
print(f)
print(fp)
print("into:")
print(NewtonsMethod)
print("\n")
fsubbed = subsfunction(flt,NewtonsMethod,"f")
print(fsubbed)
mapclstr=subsfunction(fprimelt,fsubbed,"fprime")

print(mapclstr)
os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

A=np.linspace(3,6,10,dtype=np.float32)
res_g = cl.array.to_device(queue, A)
mapcl = ElementwiseKernel(ctx,"float *X,int N,float precision",mapclstr,"mapcl")
mapcl(res_g,np.intc(100),np.float32(0.0000001)) 
print(res_g.get())



















""" for j in range(1,powerindex):
        print(fs[i][powerindex-j:powerindex])
        
        if not basefound and isvalidsymbol(fs[i],powerindex-j,powerindex):
            base=fs[i][powerindex-j:powerindex]
            print("base:")
            print(base)
            basefound = True
        if  powerindex+2+j>=len(fs[i]):
            print("cannot find end of power")
            break
        if fs[i][powerindex+2+j]==" " and not argfound:
            power=fs[i][powerindex+2:powerindex+2+j]
            print("power:")
            print(power)
            powerfound=True
            
            


"""

