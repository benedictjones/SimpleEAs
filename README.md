# SimpleEAs
Python library for some simple Evolutionary Algorithms.
These follow an ask/tell work flow where one:
- new trial members/sample are requested using ask(), these need to me manually evaluated
- the fitness socores are then fed back to the object with tell() to enable a population update

Future updates will include tools:
- Page Trend test for convergence statistical analysis

## Algorithms


### DE
Differential Evolution (DE) is an easily implemented and effective optimisation algorithm for exploiting real-world parameters. 
DE is a derivative-free, stochastic, population-based, heuristic direct search method [1] which only requires a few robust control variables. 

A simple work flow might be:
```
from pyeas import DE

# call object passing in hyperparameters and bounderies for each optimisable parameter
optimizer = DE(mut=0.6,
               crossp=0.6,
               bounds=np.array([[-5,5],[-5,5]]),

for generation in range(num_gens):

    # ask for a trial population
    trial_pop = optimizer.ask(loop=generation)

    # Loop through trial pop and evaluate their fitnesses 
    solutions = []
    for trial in trial_pop:
        fitness = f(trial)
        solutions.append((value))
    
    # Tell the object the evaluated values.
    optimizer.tell(solutions, trial_pop)

```

An Example of DE/best/1/bin solving some simple problems:

<p float="left">
  <img src="https://raw.githubusercontent.com/benedictjones/pyeas/main/examples/DE_2.gif" width="49%" />
  <img src="https://raw.githubusercontent.com/benedictjones/pyeas/main/examples/DE_6hemp.gif" width="49%" />
</p>


An example of DE/ttb/1/bin solving a 5th order polynomial to fit noisy cos(x) data.

![](https://raw.githubusercontent.com/benedictjones/pyeas/main/examples/DE.gif)


### OpenAI ES
Evolutionary Strategies (ES) involve the evaluation of a population of real valued genotypes (or population members), after which the best members are kept, and others discarded.

Natural Evolutionary Strategies (NES) are a family of Evolution Strategies which iteratively update a search distribution by using an estimated gradient on its distribution parameters.
Notably, NES performs gradient accent along the natural gradient.

The OpenAI Evolutionary Stratergy (OAIES) algorithm is a type of NES [2], implemented here as vanilla gradient decent, with momentum, or with adam optimiser [3].

A simple work flow might be:
```
from pyeas import OAIES

# call object passing in hyperparameters and bounderies for each optimisable parameter
optimizer = OAIES(alpha=0.01,
                 sigma=0.002,
                 bounds=np.array([[-10,10],[-10,10]]),
                 population_size=20,
                 optimiser = 'adam',
                 seed=1)

for generation in range(num_gens):

    # ask for a pseudo-population
    trial_pop = optimizer.ask(loop=generation)

    # Loop through pseudo-pop and evaluate their fitnesses 
    solutions = []
    for trial in trial_pop:
        fitness = f(trial)
        solutions.append((value))
    
    # Tell the object the evaluated values.
    optimizer.tell(solutions, trial_pop)

    # Calc the new parent fitness, and tell again!
    parent_fit = f(optimizer.parent)
    optimizer.tellAgain(parent_fit)
    
```


An Example of OAIES solving some simple problems:

<p float="left">
  <img src="https://raw.githubusercontent.com/benedictjones/pyeas/main/examples/OAIES_2.gif" width="49%" />
  <img src="https://raw.githubusercontent.com/benedictjones/pyeas/main/examples/OAIES_6hemp.gif" width="49%" />
</p>


## Additional Functionality

### Groupings

You might not always want a population member to be a 1d array with a corresponding boundary array. e.g., 
- member: [-0.5, -1.9, 1.6, -2, 8], 
- boundaries: [ [-1,1], [-2,2], [-2,2], [-10,10], [-10,10] ]

Instead, we might want to group a member into sub arrays corresponding to a list of boundaries. We can do this using the 'groupings' argument. e.g., 
- member: [[-0.5], [-1.9, 1.6], [-2, 8]], 
- boundaries: [ [-1,1], [-2,2], [-10,10] ]
- grouping: [1, 2, 2]


## References 

[1] R. Storn and K. Price, “Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces,” Journal of Global Optimization, vol. 11, no. 4, pp. 341–359, Dec. 1997. [Online]. Available: https://doi.org/10.1023/A:1008202821328

[2] T. Salimans, J. Ho, X. Chen, S. Sidor, and I. Sutskever, “Evolution Strategies as a Scalable Alternative to Reinforcement Learning,” Sep. 2017. [Online]. Available: http://arxiv.org/abs/1703.03864

[3] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” arXiv, Jan. 2017. [Online]. Available: http://arxiv.org/abs/1412.6980
