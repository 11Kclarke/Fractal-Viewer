cdef extern from "parameterisedComplexPoly.cpp":
    float cppmult(double zre,double zim,double cre,double cim, double reals[2])

def pymult( zre,zim,cre,cim, reallist  ):
    return cppmult(  zre,zim,cre,cim, reallist )