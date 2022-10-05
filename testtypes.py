import sympy as sp
x,y=sp.symbols("x,y")

f=1j+x**3-1
print(type(f))
fl=sp.lambdify(x,f)

import inspect
fl=inspect.getsource(fl)
#print(fl)
