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


print("\n\n\n")
print("\n\n\n")
print("\n\n\n")
print("\n\n\n")




def isvalidsymbol(line,symbols):#is only a valid symbol if its a number or a variable seen in signature
    if isinstance(line,list):
        line="".join(line) 
    if line in symbols or line.isnumeric():
        return True
    else:
        return False
#must obey order of operations
operations = ["^","/","*","+","-"]
#operationsOrig = ["base^arg","base*arg","base+arg","base-arg","base/arg"]
replacements = ["dtype_pow(base,arg)","dtype_divide(base,arg)","dtype_mul(base,arg)","dtype_add(base,arg)","dtype_add(base,dtype_neg(arg))"]
outputprogram=""
outputatoms = []
def getargs(base,arg):#assuming bidmas is followed thing infront/after operator up to first space not in position 0 will be arg/base
    #eg a^b+c == (a^b) + c ^ has a,b and + has (a^b),c
    #if the order of operations fails we may end up with
    #a^(b+c)
   
    opening=0
    closing=0
    for i,val in reversed(list(enumerate(base))):
        """print("val: "+val)
        print(base[i:])
        print(not(base[i:].isspace()))
        print(val.isspace())"""
        if val==")":closing+=1
        if val=="(":opening+=1
        if opening > closing:# if opening > closing must have started in bracket as reading backwards
            
            if not "," in base[i+1:]:
                base=base[i+1:] 
            else:
                base=base[i+1:].split(",",1)[1]
            break
        if (val.isspace() and not (base[i:].isspace())): 
            base=base[i:]
            break
    opening=0
    closing=0
    for i,val in enumerate(arg):
        """print("val: "+val)
        print(arg[:i])
        print(not(arg[:i].isspace()))
        print(val.isspace())"""
        if val==")":closing+=1
        if val=="(":opening+=1
        if opening < closing:# if opening < closing must have started in bracket
            if not "," in arg[:i-1]:
                arg=arg[:i-1] 
            else: 
                arg=arg[:i-1].split(",",1)[0] 
            break
        if val.isspace() and not(arg[:i].isspace()) and len(arg[:i])>0:
            arg=arg[:i]
            break
    return base,arg

def makesig(pysig,dtype):
    pysig=pysig[pysig.find("(")+1:-2].split(",")
    symbols=[]
    for i in range(len(pysig)):
        symbols.append(pysig[i].strip())
        pysig[i]=dtype+" "+pysig[i]
    return pysig

def dtyper(operation):
    lhs=operation.find("_")
    rhs=operation.find("(")
    args=operation[rhs+1:operation.rfind(")")].split(",")#split whats between first opening bracket and last closing bracket by first comma
    assert not (args[0].isnumeric() and args[1].isnumeric())#no variable should have been simplified
    print(args[0])
    if args[0].isnumeric():
        operation = operation[:lhs+1]+"r"+operation[lhs+1:]
    if args[1].isnumeric():
        operation = operation[:rhs]+"r"+operation[rhs:]
    return operation

def translate(fl,operations=operations,outputarg = "",dtype="cdouble_"):
    print(inspect.getsource(fl))
    fsorig=inspect.getsource(fl)
    fsorig=fsorig.replace("**","^")#a special case poor solution, to deal with difficulty seperating * from **
    workingfs  = fsorig.split("\n",1)[1]
    workingfs = workingfs.replace("return"," ")
    print(workingfs)
    c=0
    k=0
    while c<len(operations):
        if operations[c] in workingfs:
            base,arg = workingfs.split(operations[c],1)
            #print(f"new loop: {k}")
            #print(workingfs+" split into: ")
            #print(base)
            #print(arg)
            base,arg = getargs(base,arg)
            #print(f"Formated Base: {base}")
            #print(f"Formated arg: {arg}")
            replacement=replacements[c].replace("dtype_",dtype).replace("base",base.strip()).replace("arg",arg.strip())
            print(replacement)
            replacement=dtyper(replacement)
            print(replacement)
            #print(f"replaced {base+operations[c]+arg}")
            #print("working fs from:"+workingfs)
            workingfs=workingfs.replace(base+operations[c]+arg,replacement)
            #print(workingfs)
            k+=1
            assert k<6
        else:
            c+=1
        
    return workingfs
xbig,y=sp.symbols("xbig,y")
f=xbig**2+xbig**3-1.0+xbig

fp = sp.diff(f)
fl=sp.lambdify((xbig),f)

fprimel=sp.lambdify(xbig,fp)
flt=translate(fl)
fprimelt = translate(fprimel)
print(f)
print(flt)
print(fp)
print(fprimelt)        
