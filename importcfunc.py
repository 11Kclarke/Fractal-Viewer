import ctypes
import pathlib
path =  r"C:\Users\kiera\Documents\university\Cprojects\fractalstuff"
lib = pathlib.Path(path)
if __name__ == "__main__":
    # Load the shared library into ctypes
    libname = path+ "\\parameterisedComplexPoly.exe"
    print(libname)
    c_lib = ctypes.CDLL(libname)