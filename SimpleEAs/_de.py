import numpy as np

from typing import Any
from typing import cast
from typing import Optional



class DE:
    """CMA-ES stochastic optimizer class with ask-and-tell interface.

    based off the style of: https://github.com/CyberAgentAILab/cmaes/blob/main/cmaes/_cma.py

    Build into a package: https://www.youtube.com/watch?v=5KEObONUkik 

    Example:

        .. code::

           import numpy as np
           from cmaes import CMA

           def quadratic(x1, x2):
               return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

           optimizer = CMA(mean=np.zeros(2), sigma=1.3)

           for generation in range(50):
               solutions = []
               for _ in range(optimizer.population_size):
                   # Ask a parameter
                   x = optimizer.ask()
                   value = quadratic(x[0], x[1])
                   solutions.append((x, value))
                   print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")

               # Tell evaluation values.
               optimizer.tell(solutions)

    Args:

        mean:
            Initial mean vector of multi-variate gaussian distributions.

        sigma:
            Initial standard deviation of covariance matrix.

        bounds:
            Lower and upper domain boundaries for each parameter (optional).

        n_max_resampling:
            A maximum number of resampling parameters (default: 100).
            If all sampled parameters are infeasible, the last sampled one
            will be clipped with lower and upper bounds.

        seed:
            A seed number (optional).

        population_size:
            A population size (optional).

        cov:
            A covariance matrix (optional).
    """

    _toggle = 0

    def __init__(
                self,
                mean: np.ndarray,
                sigma: float,
                bounds: Optional[np.ndarray] = None,
                n_max_resampling: int = 100,
                seed: Optional[int] = None,
                population_size: Optional[int] = None,
                cov: Optional[np.ndarray] = None,
                ):
        

        self._rng = np.random.RandomState(seed)


        return
    
    def ask(self) -> np.ndarray:
        """Sample a trial population which needs to be evaluated"""

        assert self._toggle == 1, "Must first evaluate current trials and tell me their fitnesses."

        for i in range(self._n_max_resampling):
            x = self._sample_solution()
            if self._is_feasible(x):
                return x
        x = self._sample_solution()
        x = self._repair_infeasible_params(x)
        
        return x
    
    def tell(self, solutions: list) -> None:
        """Tell the object the fitness values of the trial population which has been valuated"""

        # if condition returns False, AssertionError is raised:
        assert len(solutions) == self._popsize, "Must tell with popsize-length solutions."
        assert self._toggle == 0, "Must first ask (i.e., fetch) & evaluate new trials."

        return
    

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)
        return
    



    def get(self) -> np.ndarray:
        """Fetch the current parent population"""


        return
    
    