import sys
print(sys.version)
from tokenize import Exponent
import PyopenclNewtonsFractal
from matplotlib.transforms import Transform
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr

import sys

import math

from numba import *

np.warnings.filterwarnings("ignore")

@njit("complex128(complex128,complex128)",fastmath=True)
def exponential(Zn,C):
    pow=2.718
    return pow**Zn+C

@njit("complex128(complex128,complex128)",fastmath=True)
def TriginTrig(Zn,C):
    return np.cos(C+np.sin(Zn))


#very cool, but needs to start zoomed
@njit("complex128(complex128,complex128)",fastmath=True)
def reciprocalmandlebrot(Zn,C):
    return 1/(Zn**2+C)


@njit("complex128(complex128,complex128,complex128,complex128,complex128)",fastmath=True)
def generalcomplexpoly(Zn,d,a=2,b=2,power=2):
    #returns value of complex polynomial of the form z^power+a*Zn+b*j+d
    return Zn**power+a*Zn+b*1j+d

@njit("complex128(complex128,complex128)",fastmath=True)
def mandlebrot(Zn,C):
    pow=2
    return Zn**pow+C

#cool, move to right 
@njit("complex128(complex128,complex128)",fastmath=True)
def exponential(Zn,C):
    
    return np.exp(Zn+C)