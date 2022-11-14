#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.patches import Circle, PathPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.mplot3d.art3d as art3d
import scipy
import random
from copy import deepcopy

class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})

    def show(self):
        plt.show()

    def plot_2d_function(self, fun):
        X = np.arange(-3, 3, 0.1)
        Y = np.arange(-3, 3, 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = fun(X, Y)

        surf = self.ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # Customize the z axis.
        self.ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        self.ax.zaxis.set_major_formatter('{x:.02f}')
        # Add a color bar which maps values to colors.
        self.fig.colorbar(surf, shrink=0.5, aspect=5)


    def plot_point(self, x, y, z = 0, color = None):
        circle = Circle((x, y), 0.1)
        if not color:
            if z > 0:
                color = 'blue'
            else:
                color = 'red'
        circle.set_facecolor(color)
        self.ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=z, zdir="z")


def single_modal_function(x, y):
    return np.exp(-x**2-y**2)*(1+5*x + 6*y + 12*x*np.cos(y));


def multi_modal_function(x,y):
    return 3*(1-x)**2*np.exp(-x**2 - (y+1)**2) - 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2) - 1/3*np.exp(-(x+1)**2 - y**2)


def objective_function(x, y):
    #return single_modal_function(x,y)
    return multi_modal_function(x,y)


def fitness_function(x):
    y = objective_function(*x)
    return y


def print_result(x_min, y_min, z_min):
    print('-----')
    print('xmin: %s' % (str(x_min)))
    print('ymin: %s' % (str(y_min)))
    print('zmin: %s' % (str(z_min)))
    print('-----')

# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------

def main():

    # get scipy.optimize.minimize for result for reference
    res = scipy.optimize.minimize(fitness_function, x0 = (1, -1), method='Nelder-Mead')
    x_min_0 = res.x[0]
    y_min_0 = res.x[1]
    z_min_0 = objective_function(x_min_0, y_min_0)

    print('reference result from scipy.optimize.minimize: ')
    print_result(x_min_0, y_min_0, z_min_0)


    # ------------------------------
    # custom optimization code here!
    # ==============================


    # ==============================


    plotter = Plotter()
    plotter.plot_2d_function(objective_function)
    plotter.plot_point(x_min_0, y_min_0, z_min_0)
    plotter.show()


if __name__=="__main__":
    main()
