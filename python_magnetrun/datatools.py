"""
Locally Weighted Linear Regression (Loess)

see:
https://xavierbourretsicotte.github.io/loess.html
"""

from math import ceil
import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
##from IPython.display import Image
##from IPython.display import display
#plt.style.use('seaborn-white')
## if jupyter: %matplotlib inline

import os

#Defining the bell shaped kernel function - used for plotting later on
def kernel_function(xi,x0,tau= .005):
    return np.exp( - (xi - x0)**2/(2*tau)   )

def lowess_bell_shape_kern(x, y, tau = .005):
    """lowess_bell_shape_kern(x, y, tau = .005) -> yest
    Locally weighted regression: fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The kernel function is the bell shaped function with parameter tau. Larger tau will result in a
    smoother curve. 
    """
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function
    w = np.array([np.exp(- (x - x[i])**2/(2*tau)) for i in range(n)])

    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                      [np.sum(weights * x), np.sum(weights * x * x)]])
        theta = linalg.solve(A, b)
        yest[i] = theta[0] + theta[1] * x[i]

    return yest

def lowess_ag(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)

    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

            residuals = y - yest
            s = np.median(np.abs(residuals))
            delta = np.clip(residuals / (6.0 * s), -1, 1)
            delta = (1 - delta ** 2) ** 2
            
    return yest

import statsmodels.api as sm
def lowess_sm(x, y, f=1. / 3., iter=3):
    """
    ref:
    https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    """
    lowess_sm_ = sm.nonparametric.lowess
    return lowess_sm_(y, x, f, iter, return_sorted = False)

def smooth(mrun, key, inplace, tau, debug, show, input_file):
    """
    """
    
    filteredkey = "filtered%s" % key
    df = mrun.getData() #.extractData(keys)
    Meanval = df[key].mean()
    
    #Initializing noisy non linear data
    x = df["t"].to_numpy() #np.linspace(0,1,100)
    y = df[key].to_numpy() # np.sin(x * 1.5 * np.pi )

    if debug:
        plt.figure(figsize=(10,6))
        plt.scatter(x,y, facecolors = 'none', edgecolor = 'darkblue', label = key)

    #try:
    yest_bell = lowess_bell_shape_kern(x,y,tau)
    plt.plot(x,yest_bell,color = 'red', label = 'Loess: bell shape kernel')
    if debug:
        x0 = (x[0] + x[40])/2.
        plt.fill(x[:40],Meanval*kernel_function(x[:40],x0,tau), color = 'lime', alpha = .5, label = 'Bell shape kernel')
        if show:
            plt.show()
        else:
            f_extension = os.path.splitext(input_file)[-1]
            imagefile = input_file.replace(f_extension, "-smoothed%s.png" % key)
            print("save to %s" % imagefile)
            plt.savefig(imagefile, dpi=300)
            plt.close()

    if inplace:
        # replace key by filtered ones
        df[filteredkey] = pd.Series(yest_bell)
        del df[key]
        df.rename(columns={filteredkey: key}, inplace=True)
    # except:
    #     print("smooth: bell_shape_kern smoother failed")
        
def filterpikes(mrun, key, inplace, threshold, twindows, debug, show, input_file):
    """
    # filter spikes 
    # see: https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/
    """

    print("type(mrun):", type(mrun))
    df = mrun.getData() #.extractData(keys)

    kw = dict(marker='o', linestyle='none', color='r', alpha=0.3)
    mean = df[key].mean()
    # print("%s=" % key, type(df[key]), df[key])

    if mean != 0:
        var = df[key].var()
        # print("mean(%s)=%g" % (key,mean), "std=%g" % math.sqrt(var) )
        filtered = df[key].rolling(window=twindows, center=True).median().fillna(method='bfill').fillna(method='ffill')
        filteredkey = "filtered%s" % key
        # print("** ", filteredkey, type(filtered))
            
        df = df.assign(**{filteredkey: filtered.values})
        # print("*** ", filteredkey, df[filteredkey])
            
        difference = np.abs((df[key] - filtered)/mean*100) 
        outlier_idx = difference > threshold
        
        if debug:
            ax = plt.gca()
            df[key].plot()
            # filtered.plot()
            df[filteredkey].plot()
            df[key][outlier_idx].plot(**kw)
            
            ax.legend()
            plt.grid(b=True)
            plt.title(mrun.getInsert().replace("_","\_") + ": Filtered %s" % key)
            if show:
                plt.show()
            else:
                f_extension = os.path.splitext(input_file)[-1]
                imagefile = input_file.replace(f_extension, "-filtered%s.png" % key)
                print("save to %s" % imagefile)
                plt.savefig(imagefile, dpi=300)
                plt.close()

        if inplace:
            # replace key by filtered ones
            del df[key]
            df.rename(columns={filteredkey: key}, inplace=True)

if __name__ == "__main__":

    import argparse
    import python_magnetrun
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--keys", help="specify keys to select (eg: Tin1;Tin2)", default="Tin1")
    parser.add_argument("--smoothing_f", help="specify smoothing f param", type=float, default=0.25)
    parser.add_argument("--smoothing_tau", help="specify smoothing tau param", type=float, default=0.005)
    parser.add_argument("--smoothing_iter", help="specify smoothing iter param", type=int, default=5)
    parser.add_argument("--show", help="display graphs (requires X11 server active)", action='store_true')
    args = parser.parse_args()
    
    f_extension=os.path.splitext(args.input_file)[-1]
    if f_extension != ".txt":
        print("so far only txt file support is implemented")
        sys.exit(0)

    filename = os.path.basename(args.input_file)
    result = filename.startswith("M")
    if result:
        try:
            index = filename.index("_")
            args.site = filename[0:index]
            print("site detected: %s" % args.site)
        except:
            print("no site detected - use args.site argument instead")
            pass
    mrun = python_magnetrun.MagnetRun.fromtxt(args.site, args.input_file)
    mrun.getMData().addTime()
    start_timestamp = mrun.getMData().getStartDate()
    dkeys = mrun.getKeys()

    skeys = args.keys.split(";")
    for key in skeys:
        selected_df = mrun.getMData().extractData(["t", key])
        Meanval = selected_df[key].mean()
    
        #Initializing noisy non linear data
        x = selected_df["t"].to_numpy() #np.linspace(0,1,100)
        #noise = np.random.normal(loc = 0, scale = .25, size = 100)
        y = selected_df[key].to_numpy() # np.sin(x * 1.5 * np.pi )
        #y_noise = y + noise

        print("display Weighted Linear Regression")
        plt.figure(figsize=(10,6))
        #plt.plot(x,y,color = 'darkblue', label = 'sin()')
        plt.scatter(x,y, facecolors = 'none', edgecolor = 'darkblue', label = key)

        #
        print("compute Locally Weighted Linear Regression")
        try:
            f = args.smoothing_f #25
            yest = lowess_ag(x, y, f=f, iter=args.smoothing_iter)
            plt.plot(x,yest, color = 'orange', label = 'Loess: A. Gramfort')
        except:
            print("Failed to build lowess_ag")


        try:
            tau = args.smoothing_tau
            yest_bell = lowess_bell_shape_kern(x,y,tau)
            # x0 = (x[0] + x[40])/2.
            # plt.fill(x[:40],Meanval*kernel_function(x[:40],x0,args.smoothing_tau), color = 'lime', alpha = .5, label = 'Bell shape kernel')
            plt.plot(x,yest_bell,color = 'red', label = 'Loess: bell shape kernel')
        except:
            print("Failed to build bell")

        try:
            f = 0.7 # between 0 and 1, by default: 2./3.
            yest_sm = lowess_sm(x,y, f, 3) # iter from 3 to 6; delta = 0.01 * range(x)) #args.smoothing_iter
            plt.plot(x,yest_sm, color = 'magenta', label = 'Loess: statsmodel') # marker="o", 
        except:
            print("Failed to build sm")

        plt.legend()
        plt.title('Loess regression comparisons')
        if args.show:
            plt.show()
        else:
            imagefile = mrun.getSite()
            start_date = ""
            start_time = ""
            if "Date" in dkeys and "Time" in dkeys:
                tformat="%Y.%m.%d %H:%M:%S"
                start_date=mrun.getMData().getData("Date").iloc[0]
                start_time=mrun.getMData().getData("Time").iloc[0]
                
                plt.savefig('%s_%s---%s-smoothed-%s.png' % (imagefile,str(start_date),str(start_time),key) , dpi=300 )
                plt.close()
