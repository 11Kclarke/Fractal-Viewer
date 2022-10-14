import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
import os
import matplotlib.pyplot as plt
import sympy as sp
import timeit
import Main
import PyToPyOpenCL#mine


os.environ["PYOPENCL_CTX"]="0"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

stabilityfractal="""
    int C = 0;
    float thresh =2;
    dtype_t lastval;

    fval.real=100;
    dtype_t lastval;
    lastval.real=0;
    lastval.imag=0;
    while (dtype_abs(fval)>thresh && C<N) 
    {
        fval=__f(X[i],lastval)__;
        
        X[i]= dtype_add(X[i],dtype_neg(dtype_divide(fval,fpval)));
        C+=1;
    }
    """