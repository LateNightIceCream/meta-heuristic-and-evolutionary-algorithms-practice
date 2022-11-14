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

# -----------------------------------------------------------------------------

def exploratory_move_GPS(base_point, mesh_size):
    points = []
    i = 0
    for decision_variable in base_point:

        step = np.zeros(len(base_point))
        step[i] = 1
        step *= mesh_size

        p1 = base_point + step
        p2 = base_point - step

        points.append(p1)
        points.append(p2)

        i += 1

    points.append(base_point)

    return points

    # example for 2d base
    # return [
    #     (base_x, base_y),
    #     (base_x + mesh_size, base_y),
    #     (base_x - mesh_size, base_y),
    #     (base_x, base_y + mesh_size),
    #     (base_x, base_y + mesh_size)
    # ]

def exploratory_move_MADS(base_point, mesh_size):
    # positive in all directions each
    # + one with all in negative direction
    points = []
    i = 0
    for decision_variable in base_point:

        step = np.zeros(len(base_point))
        step[i] = 1
        step *= mesh_size

        p1 = base_point + step

        points.append(p1)

        i += 1

    points.append(base_point)
    points.append(base_point - mesh_size)

    return points



def exploratory_move(base_point, mesh_size):
    #return exploratory_move_GPS(base_point, mesh_size)
    return exploratory_move_MADS(base_point, mesh_size)


def eval_exploration(points, fitness_function):
    best_fitness = float('inf')
    best_point = float('inf')
    for point in points:
        fitness = fitness_function(point)
        if fitness < best_fitness:
            best_fitness = fitness
            best_point = point

    return (best_fitness, best_point)


def pattern_move(current_point, previous_point, acceleration_factor):
    direction = current_point - previous_point
    return np.add(previous_point, acceleration_factor * direction)


def pattern_search_minimize(fitness_function, exploratory_move = exploratory_move, starting_point = None):
    if not isinstance(starting_point, np.ndarray):
        base_point = np.array([random.uniform(-3, 3), random.uniform(-3, 3)])
    else:
        base_point = starting_point

    print("initial base point:")
    print(base_point)

    initial_mesh_size = 0.05
    mesh_contraction = 0.1
    mesh_size = initial_mesh_size

    acceleration_factor = 1 # for pattern move

    fitness_change = float('inf')
    fitness_change_termination_threshold = 1e-5
    previous_fitness = fitness_function(base_point)
    current_fitness = None
    current_x = None

    moves = []

    while abs(fitness_change) > fitness_change_termination_threshold:
        print("ok")

        # exploratory move
        print("exploratory move")

        exploration = exploratory_move(base_point, mesh_size)

        current_fitness, best_explored = eval_exploration(exploration, fitness_function)

        if current_fitness < previous_fitness:

            # pattern move
            print("pattern move")

            # reset mesh size
            mesh_size = initial_mesh_size
            current_point = best_explored

            while current_fitness < previous_fitness:
                previous_fitness = current_fitness
                previous_point = base_point
                base_point = current_point

                pattern_move_point = pattern_move(base_point, previous_point, acceleration_factor)

                best_fitness, best_explored = eval_exploration(exploratory_move(pattern_move_point, mesh_size), fitness_function)

                if best_fitness < current_fitness:
                    current_fitness = best_fitness
                    current_point = best_explored
                    moves.append(best_explored)

        else: # no point in the exploration is better than the base point
            # decrese mesh size
            print("contracting mesh!")
            mesh_size -= mesh_contraction


        fitness_change = current_fitness - previous_fitness
        print(current_fitness)
        print(previous_fitness)
        print(fitness_change)

    # terminated

    print("final base_point:")
    print(base_point)

    return moves

# -----------------------------------------------------------------------------

def main():

    # get scipy.optimize.minimize for result for reference
    res = scipy.optimize.minimize(fitness_function, x0 = (1, -1), method='Nelder-Mead')
    x_min_0 = res.x[0]
    y_min_0 = res.x[1]
    z_min_0 = objective_function(x_min_0, y_min_0)

    print('reference result from scipy.optimize.minimize: ')
    print_result(x_min_0, y_min_0, z_min_0)

    moves = pattern_search_minimize(fitness_function)
    print(moves)
    x_min_1 = 0
    y_min_1 = 0
    z_min_1 = 0


    # comparison between GPS and MADS
    #starting_pos = np.array([random.uniform(-3, 3), random.uniform(-3, 3)])
    starting_pos = np.array([1.2, -0.3])
    moves_gps = pattern_search_minimize(fitness_function, exploratory_move_GPS, starting_pos)
    moves_mads = pattern_search_minimize(fitness_function, exploratory_move_MADS, starting_pos)

    points = exploratory_move(np.array([1,1]), 0.3)

    print('result from custom optimization:')
    print_result(x_min_1, y_min_1, z_min_1)

    plotter = Plotter()
    plotter.plot_2d_function(objective_function)
    plotter.plot_point(x_min_0, y_min_0, z_min_0)
    #plotter.plot_point(x_min_1, y_min_1, z_min_1, color = 'lime')

    # for point in points:
    #     plotter.plot_point(point[0], point[1], 0)

    for point in moves_gps:
        plotter.plot_point(point[0], point[1], fitness_function(point), color='lime')

    for point in moves_mads:
        plotter.plot_point(point[0], point[1], fitness_function(point), color='pink')


    plotter.show()


if __name__=="__main__":
    main()


