import sys
print(sys.version)
import numpy as np
import pandas as pd
import datashader as ds
from datashader import transfer_functions as tf
import numba
from numba import jit,njit,prange

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from datashader.mpl_ext import dsshow, alpha_colormap
import timeit




"""Stolen from https://www.architecture-performance.fr/ap_blog/plotting-hopalong-attractor-with-datashader-and-numba/"""
@njit(fastmath=True)
def hopalong_1(x, y, a, b, c):
    return y - np.sqrt(np.fabs(b * x - c)) * np.sign(x), a - x


"""Stolen from https://www.architecture-performance.fr/ap_blog/plotting-hopalong-attractor-with-datashader-and-numba/"""
@njit(fastmath=True)
def hopalong_2(x, y, a, b, c):
    return y - 1.0 - np.sqrt(np.fabs(b * x - 1.0 - c)) * np.sign(x - 1.0), \
           a - x - 1.0

"""Stolen from https://www.architecture-performance.fr/ap_blog/plotting-hopalong-attractor-with-datashader-and-numba/"""
@njit
def AttractorIterator(fn, a, b, c, x0=0, y0=0, n=50):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in np.arange(n-1):
        x[i+1], y[i+1] = fn(x[i], y[i], a, b, c)
    
    return (x,y)

width = 500
height = 500
"""Stolen from https://www.architecture-performance.fr/ap_blog/plotting-hopalong-attractor-with-datashader-and-numba/"""
@jit
def compute_and_plot(fn, a, b, c,x0=0,y0=0,n=50):
    cvs = ds.Canvas(plot_width=width, plot_height=height)
    trajectory =AttractorIterator(fn, a, b, c,x0=x0,y0=y0,n=n)
    #print(type(trajectory))
    df = pd.DataFrame(dict(x=trajectory[0],y=trajectory[1]))
    
    #print(type(cvs.points(df, 'x', 'y').values))
    #print(type(cvs.points(df, 'x', 'y')))
    return cvs.points(df, 'x', 'y').values
    #return tf.shade(agg, cmap=purples)
    return df
@jit(parallel =True, nogil = True)
def rangeimage(res,extent,attractor=hopalong_1,iterator=compute_and_plot,n=50):
    starttime = timeit.default_timer()

    valsx=np.linspace(extent[0],extent[1],res)
    valsy=np.linspace(extent[2],extent[3],res)
    resultrange=np.zeros((res,res))
    for xy in prange(res*res):
         agg = iterator(attractor, 2.0, 1.0, 0.0,x0=valsx[xy%res],y0=valsy[xy//res],n=n)
         #print(agg.shape)
         resultrange[xy%res,xy//res] = np.sum(np.ptp(agg,axis=1))+np.sum(np.ptp(agg,axis=0))
         #resultrange[xy%res,xy//res] =np.sum(abs(np.diff(agg)))
         if xy%10000 == 0 and not xy==0:
            print("Has generated:")
            print(str(xy)+"/"+str(res**2))
            print("in "+str(timeit.default_timer()-starttime)+" (s)")
            s=(res**2/xy)*(timeit.default_timer()-starttime)
            print("estimated "+str(s//60)+" Minutes and "+str(s%60)+" seconds remaining")
    return resultrange

if __name__ == '__main__':
    # image size
    
    purples = plt.get_cmap('Purples')
    # number of steps
    res=200
    extent = (-2,2,-2,2)
    fig, ax = plt.subplots()
    n = 100
    #img=rangeimage(res,extent,n=n)
    img=compute_and_plot()
    ax.imshow(img,extent=extent)
    """fig, ax = plt.subplots(2,2)
    n = 120
    img=rangeimage(res,extent,n=n)
    ax[0,0].imshow(img,extent=extent)
    n = 240
    img2=rangeimage(res,extent,n=n)
    ax[1,0].imshow(img2,extent=extent)
    n = 480
    img3=rangeimage(res,extent,n=n)
    ax[0,1].imshow(img3,extent=extent)
    n=900
    img4=rangeimage(res,extent,n=n)
    ax[1,1].imshow(img4,extent=extent)"""
    #ax.set_xlabel("X0")
    #ax.set_ylabel("Y0")
    #ax.spines['left'].set_position('center')
    #ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    #ax.spines['right'].set_color('none')
    #ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    #ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('left') 
    #axs[0].imshow(resultmean)
    #axs[1].imshow(resultrange)
    #axs[2].plot(resultmedian)
    #axs[2].imshow(resultmedian)
    plt.show()
    