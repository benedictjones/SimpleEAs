import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pyeas._de import DE
from pyeas._oaies import OAIES
from pyeas._pagetest import PageTest



def f_mat(x1, x2):
    """  Matyas function: https://www.sfu.ca/~ssurjano/matya.html
            Domian: [[-10,10],[-10,10]]"""
    return 0.26*(x1**2 + x2**2) - 0.48*x1*x2

def f_bohach(x1, x2):
    """Bohachevsky Function: https://www.indusmic.com/post/bohachevsky-function
    Domian: [[-100,100],[-100,100]]"""
    return x1**2 +2*(x2**2)-0.3*np.cos(3*np.pi*x1)-0.4*np.cos(4*np.pi*x2)+0.7

def f_3hc(x1, x2):
    """" Three hump camel functions:
            domain: [-5,5],[-5,5] 
    https://www.indusmic.com/post/three-hump-camel-function
    """
    return 2*(x1**2)-1.05*(x1**4)+((x1**6)/6)+(x1*x2)+(x2**2)


def f_6hc(x1, x2):
    """ Six hump camel functions: https://www.indusmic.com/post/six-hump-camel-function
            Domain: [-3,3],[-2,2] 
            The function has global minimum f (x*) = -1.0316, at x*= (0.0898,-0.7126) and (-0.0898, 0.7126)."""
    return 4*x1**2-2.1*x1**4+(x1**6)/3+x1*x2-4*x2**2+4*x2**4

def f_kean(x1,x2):
    """ Keane Function: https://www.indusmic.com/post/python-implementation-of-keane-function

    Input Domain:
        The Keane Function is defined on input range x  [0,10] and y [0,10].

    Global Minima :
        The Keane Function has two global minimum f(x*) = 0.673667521146855 at
            x* = (1.393249070031784, 0)
            x* = (0, 1.393249070031784)
    """
    a=-np.sin(x1-x2)**2*np.sin(x1+x2)**2
    b=np.sqrt(x1*x1+x2*x2)   
    c=a/b
    return c 

def f_ackley(x, y):
    """ Ackley function:
            Domain: [-5,5],[-5,5]
    """
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20
 
def f_rose(x1, x2):
    """ Rosenbrock function: https://www.indusmic.com/post/rosenbrock-function
            domain: [-5,10],[-5,10] 
    """
    return 100*(x2-x1**2)**2+(x1-1)**2


def f_beale(x, y):
    """ Rosenbrock function: https://www.sfu.ca/~ssurjano/beale.html
            domain: [-4.5,4.5],[-4.5,4.5] 
    """
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2





"""
# #####################################
Performing DE and OAIES on several problems.
    > Many problems taken from: https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization 

This then allows for a page trend test.
"""

itrbl = [
        [f_mat, [[-10,10],[-10,10]], 'matyas'],
        [f_bohach, [[-100,100],[-100,100]], 'bohachevsky'],
        [f_3hc, [[-5,5],[-5,5]], '3hc'],
        [f_6hc, [[-3,3],[-2,2]], '6hc'],
        [f_kean, [[-10,10],[-10,10]], 'kean'],
        [f_ackley, [[-5,5],[-5,5]], 'ackley'],
        [f_rose, [[-5,10],[-5,10]], 'rosen'],
        [f_beale, [[-4.5,4.5],[-4.5,4.5]],'beale']
        ]

funs, bds, labs = zip(*itrbl)

PT = PageTest(num_cuts=5,
              problem_labels=labs,
              invert=0)


num_gens = 40
num_loops = 20

for deets in itrbl:

    fun, bound, lab = deets
    print("\n>>", lab, bound)

    


    # # Perform DE (Algorithm A)
    de_rep_fits = []
    for rep in range(num_loops):
        optimizer = DE(mut=0.6,
                    crossp=0.6,
                    bounds=np.array(bound),
                    population_size=20,
                    mut_scheme = 'best1',  # 'ttb1', rand1
                    seed=rep)

        for generation in range(num_gens):
            solutions = []
            trial_pop = optimizer.ask(loop=generation)
            for j, trial in enumerate(trial_pop):
                value = fun(trial[0], trial[1])
                solutions.append((value))
            optimizer.tell(solutions, trial_pop)

        de_rep_fits.append(optimizer.history['best_fits'])

    

    # # Perform OpenAi-ES (Algorithm B)
    oaies_rep_fits = []
    for rep in range(num_loops):
        optimizer = OAIES(
                    alpha=0.01,
                    sigma=0.001,
                    bounds=np.array(bound),
                    population_size=20,
                    optimiser = 'adam',
                    seed=num_loops+rep)

        for generation in range(num_gens):
            solutions = []
            trial_pop = optimizer.ask(loop=generation)
            for j, trial in enumerate(trial_pop):
                value = fun(trial[0], trial[1])
                solutions.append((value))
            optimizer.tell(solutions, trial_pop, t=generation)
            parent_fit = fun(optimizer.parent[0], optimizer.parent[1])
            optimizer.tellAgain(parent_fit)

        oaies_rep_fits.append(optimizer.history['best_fits'])



    PT.add_problem(x=np.arange(num_gens),
                   yA=np.mean(de_rep_fits, axis=0), # Algorithm A
                   yB=np.mean(oaies_rep_fits, axis=0) # Algorithm B
                   )



PT.test()
PT.plot_cuts()

