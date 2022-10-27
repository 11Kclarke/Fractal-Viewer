# Fractal-Viewer
Python Program to View and Zoom Different Fractals

Main contains front end and can be used as script for zooming and generating newton type and stability fractals
The Newton type generator, and create fractals from any function with more then 3 roots, that can be found with newtons method.
The Stability type fractal, finds stability of a 2 d function in the plane of input values. Mandlebrot set would be an example of stability fractals.

Current implementation uses PyOpencl to iterate fractal on gpu
PyToOpencl contains function to add function substitution to PyOpencl, as well as to convert sympy expressions to Opencl code
