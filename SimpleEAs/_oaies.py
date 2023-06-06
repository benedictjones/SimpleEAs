import numpy as np

from typing import Any
from typing import cast
from typing import Optional



class OAIES:
    """CMA-ES stochastic optimizer class with ask-and-tell interface.

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
        




        return